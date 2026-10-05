"""Recover the model's reasoning for every trajectory step, whatever the engine reports.

Engines do not always surface reasoning in their event stream: Codex emits none for
OpenRouter models (the raw ``reasoning_text`` only travels over the API), and only
summaries for OpenAI models. The capture proxy, however, records every model response
verbatim (``raw_dumps/response_NNN.txt``), and Codex's own rollout transcript keeps the
response items too. This module turns either source into an ordered list of *response
records* (reasoning + the actions that response produced) and attaches each record's
reasoning to the trajectory step that carries its action.

Each step that gets reasoning also gets ``extra.reasoning_kind``:
  ``raw``       full chain of thought text (open-weight models via OpenRouter, Claude thinking)
  ``summary``   provider-written summary (OpenAI / Gemini reasoning summaries)
  ``encrypted`` the provider returned only an encrypted blob; no readable text exists
                (``reasoning_content`` stays empty; the blob remains in raw_dumps)
Steps whose reasoning came from the engine stream itself are left untouched.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_KIND_RANK = {"raw": 3, "summary": 2, "encrypted": 1}


@dataclass
class ResponseRecord:
    """One model response: its reasoning and the actions it produced."""

    request_index: int
    reasoning: list[tuple[str, str]] = field(default_factory=list)  # (kind, text)
    anchors: list[str] = field(default_factory=list)  # normalized action texts
    has_action: bool = False
    n_actions: int = 0  # tool calls + assistant messages in this response

    @property
    def kind(self) -> str | None:
        kinds = [k for k, _ in self.reasoning]
        return max(kinds, key=lambda k: _KIND_RANK[k]) if kinds else None

    @property
    def text(self) -> str:
        return "\n\n".join(t for k, t in self.reasoning if k != "encrypted" and t.strip())


# ---------------------------------------------------------------------------
# Normalization / matching
# ---------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")


def _norm(text: str) -> str:
    """Quoting/escaping-insensitive form for matching an API action to a step."""
    return _NON_ALNUM.sub("", text or "").lower()


def _strings_in(obj: Any) -> list[str]:
    if isinstance(obj, str):
        try:  # API arguments are often a JSON string
            parsed = json.loads(obj)
        except (ValueError, TypeError):
            return [obj]
        return _strings_in(parsed) if not isinstance(parsed, str) else [parsed]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings_in(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings_in(v)]
    return []


_SHELL_WRAP = re.compile(r"^\s*(?:/usr)?/bin/(?:ba|z)?sh\s+-l?c\s+(?P<q>['\"]?)(?P<body>.*)(?P=q)\s*$", re.S)


# Arguments that carry an action's content. Auxiliary ones (workdir, timeouts, ...) are
# shared across many actions and would make unrelated steps look alike.
_PRIMARY_ARGS = ("cmd", "command", "input", "patch", "chars", "code", "content", "plan",
                 "explanation", "query", "message", "text", "path", "file_path")


def _anchors(obj: Any) -> list[str]:
    """Normalized, individually matchable strings of an API action's primary arguments."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except (ValueError, TypeError):
            pass
    if isinstance(obj, dict):
        primary = {k: v for k, v in obj.items() if k in _PRIMARY_ARGS}
        if primary:
            obj = primary
    return [a for a in (_norm(s) for s in _strings_in(obj)) if a]


def _step_strings(step: dict) -> list[str]:
    """Normalized strings a step exposes: its message, every tool-call string argument,
    and shell commands with the engine's ``/bin/bash -lc '...'`` wrapper removed."""
    out = [_norm(step.get("message") or "")]
    for tc in step.get("tool_calls") or []:
        for s in _strings_in(tc.get("arguments")):
            out.append(_norm(s))
            m = _SHELL_WRAP.match(s)
            if m:
                out.append(_norm(m.group("body")))
    return [x for x in out if x]


def _matches(record: ResponseRecord, step_strings: list[str]) -> bool:
    """Full-content match (exact, or containment for strings >= 12 chars), so long shared
    prefixes such as ``cd /long/work/dir && ...`` cannot cause false matches."""
    for a in record.anchors:
        for b in step_strings:
            if a == b or (len(a) >= 12 and a in b) or (len(b) >= 12 and b in a):
                return True
    return False


# ---------------------------------------------------------------------------
# Record sources
# ---------------------------------------------------------------------------

def _sse_events(path: Path) -> list[dict]:
    events: list[dict] = []
    for block in path.read_text(errors="replace").split("\n\n"):
        for line in block.strip().split("\n"):
            if line.startswith("data:"):
                data = line[5:].strip()
                if data and data != "[DONE]":
                    try:
                        events.append(json.loads(data))
                    except json.JSONDecodeError:
                        pass
    return events


def _texts(blocks: Any) -> str:
    if isinstance(blocks, str):
        return blocks
    return "".join(b.get("text", "") for b in blocks or [] if isinstance(b, dict))


def _record_from_responses_item(rec: ResponseRecord, item: dict) -> None:
    t = item.get("type")
    if t == "reasoning":
        raw = _texts(item.get("content"))
        summ = _texts(item.get("summary"))
        if raw.strip():
            rec.reasoning.append(("raw", raw))
        elif summ.strip():
            rec.reasoning.append(("summary", summ))
        elif item.get("encrypted_content"):
            rec.reasoning.append(("encrypted", ""))
    elif t in ("function_call", "custom_tool_call", "local_shell_call"):
        rec.has_action = True
        rec.n_actions += 1
        rec.anchors.extend(_anchors(item.get("arguments") or item.get("input") or item.get("action") or ""))
    elif t == "message" and item.get("role", "assistant") == "assistant":
        txt = _texts(item.get("content"))
        if txt.strip():
            rec.has_action = True
            rec.n_actions += 1
            rec.anchors.append(_norm(txt))


def parse_responses_sse(path: Path, request_index: int) -> ResponseRecord:
    """OpenAI Responses API stream (Codex) -> record (output items in order)."""
    rec = ResponseRecord(request_index=request_index)
    events = _sse_events(path)
    done = [e["item"] for e in events if e.get("type") == "response.output_item.done" and e.get("item")]
    if not done:  # fall back to the terminal response object
        final = next((e.get("response") for e in reversed(events)
                      if e.get("type") in ("response.completed", "response.incomplete")), None)
        done = (final or {}).get("output") or []
    for item in done:
        _record_from_responses_item(rec, item)
    return rec


def parse_anthropic_sse(path: Path, request_index: int) -> ResponseRecord:
    """Anthropic Messages stream (Claude Code) -> record."""
    rec = ResponseRecord(request_index=request_index)
    blocks: dict[int, dict] = {}
    for e in _sse_events(path):
        t = e.get("type")
        if t == "content_block_start":
            b = dict(e.get("content_block") or {})
            b.setdefault("_text", "")
            blocks[e.get("index", len(blocks))] = b
        elif t == "content_block_delta":
            b = blocks.setdefault(e.get("index", 0), {"type": "?", "_text": ""})
            d = e.get("delta") or {}
            b["_text"] += d.get("thinking") or d.get("text") or d.get("partial_json") or ""
    for _, b in sorted(blocks.items()):
        bt = b.get("type")
        if bt == "thinking" and b["_text"].strip():
            rec.reasoning.append(("raw", b["_text"]))
        elif bt == "redacted_thinking":
            rec.reasoning.append(("encrypted", ""))
        elif bt == "tool_use":
            rec.has_action = True
            rec.n_actions += 1
            rec.anchors.extend(_anchors(b["_text"] or b.get("input") or ""))
        elif bt == "text" and b["_text"].strip():
            rec.has_action = True
            rec.n_actions += 1
            rec.anchors.append(_norm(b["_text"]))
    return rec


def records_from_raw_dumps(session_dir: Path) -> list[ResponseRecord]:
    """Main-agent response records from the capture proxy's raw dumps, in request order."""
    raw = session_dir / "raw_dumps"
    if not raw.is_dir():
        return []
    contexts: dict[int, str] = {}
    cap = session_dir / "api_captures.jsonl"
    if cap.exists():
        for line in cap.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("request_index") is not None:
                contexts[e["request_index"]] = e.get("agent_context") or "main"
    out: list[ResponseRecord] = []
    for p in sorted(raw.glob("response_*.txt")):
        m = re.fullmatch(r"response_(\d+)\.txt", p.name)
        if not m:
            continue
        idx = int(m.group(1))
        if contexts.get(idx, "main") != "main":
            continue  # subagent / SDK-internal calls belong to other trajectories
        req = raw / f"request_{idx:03d}.json"
        path_hint = ""
        hdr = raw / f"request_{idx:03d}_headers.json"
        if hdr.exists():
            try:
                path_hint = json.loads(hdr.read_text()).get("path", "")
            except json.JSONDecodeError:
                pass
        is_responses = "/responses" in path_hint or (
            not path_hint and req.exists() and '"input"' in req.read_text(errors="replace")[:4000])
        out.append(parse_responses_sse(p, idx) if is_responses else parse_anthropic_sse(p, idx))
    return out


def records_from_codex_rollout(transcript: Path) -> list[ResponseRecord]:
    """Fallback when no proxy capture exists: group the rollout's response items into
    responses (reasoning + following actions, closed by the next tool output/user turn)."""
    out: list[ResponseRecord] = []
    cur: ResponseRecord | None = None
    n = 0
    for line in transcript.read_text(errors="replace").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") != "response_item":
            continue
        item = e.get("payload") or {}
        t = item.get("type")
        closes = t in ("function_call_output", "custom_tool_call_output") or (
            t == "message" and item.get("role") in ("user", "developer"))
        if closes:
            if cur is not None:
                out.append(cur)
                cur = None
            continue
        if cur is None or (t == "reasoning" and cur.has_action):
            if cur is not None:
                out.append(cur)
            n += 1
            cur = ResponseRecord(request_index=n)
        _record_from_responses_item(cur, item)
    if cur is not None:
        out.append(cur)
    return out


# ---------------------------------------------------------------------------
# Attachment
# ---------------------------------------------------------------------------

def _is_action_step(step: dict) -> bool:
    if step.get("source") != "agent":
        return False
    tcs = step.get("tool_calls") or []
    if any(tc.get("function_name") != "error" for tc in tcs):
        return True
    return bool((step.get("message") or "").strip())


def attach_reasoning(steps: list[dict], records: list[ResponseRecord], window: int = 12) -> dict:
    """Attach each record's reasoning to the step carrying its action (monotonic
    alignment by action content; unmatched reasoning-only records roll forward onto the
    next matched step so no reasoning is dropped). Steps that already have reasoning from
    the engine stream keep it. Mutates ``steps``; returns coverage stats."""
    stats = {"records": len(records), "with_reasoning": sum(1 for r in records if r.reasoning),
             "attached": 0, "steps_filled": 0, "kinds": {}, "unmatched_records": 0}
    p = 0
    used = 0  # actions of records[p - 1] already matched to steps
    action_steps = [s for s in steps if _is_action_step(s)]
    for step in action_steps:
        strs = _step_strings(step)
        if p > 0 and used < records[p - 1].n_actions and _matches(records[p - 1], strs):
            used += 1
            continue  # another action of the same response (e.g. message + call, 2 calls)
        j = next((k for k in range(p, min(len(records), p + window)) if _matches(records[k], strs)), None)
        if j is None:
            continue
        group = records[p:j + 1]
        p = j + 1
        used = 1
        _apply(step, group, stats)
    leftover = [r for r in records[p:] if r.reasoning]
    if leftover and action_steps:
        _apply(action_steps[-1], leftover, stats)
    stats["unmatched_records"] = len(records) - p
    return stats


def _apply(step: dict, group: list[ResponseRecord], stats: dict) -> None:
    with_r = [r for r in group if r.reasoning]
    if not with_r:
        return
    stats["attached"] += len(with_r)
    if (step.get("reasoning_content") or "").strip():
        return  # the engine already surfaced this step's reasoning
    text = "\n\n".join(r.text for r in with_r if r.text)
    kind = max((r.kind for r in with_r), key=lambda k: _KIND_RANK[k])
    if text:
        step["reasoning_content"] = (step.get("reasoning_content") or "") + text
    step.setdefault("extra", {})["reasoning_kind"] = kind
    stats["steps_filled"] += 1
    stats["kinds"][kind] = stats["kinds"].get(kind, 0) + 1


# Providers whose readable reasoning text is a model-written summary of hidden thinking,
# even when delivered in the raw ``reasoning_text`` field.
_SUMMARY_MODELS = re.compile(r"(^|/)(gemini|gpt-|o[1-9]|codex)", re.I)


def enrich_trajectory_reasoning(traj: dict, session_dir: Path, engine: str,
                                model: str | None = None) -> dict:
    """Fill step reasoning from the best available source. Returns coverage stats."""
    records = records_from_raw_dumps(session_dir)
    source = "api_capture"
    if not records and engine == "codex" and (session_dir / "transcript.jsonl").exists():
        records = records_from_codex_rollout(session_dir / "transcript.jsonl")
        source = "codex_rollout"
    if not records:
        return {"source": None, "records": 0}
    if model and _SUMMARY_MODELS.search(model):
        for r in records:
            r.reasoning = [("summary" if k == "raw" else k, t) for k, t in r.reasoning]
    stats = attach_reasoning(traj.get("steps") or [], records)
    stats["source"] = source
    return stats
