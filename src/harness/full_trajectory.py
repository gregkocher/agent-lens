"""One ATIF trajectory per run, for viewing and analysis (never rendered for judges).

A run's ``session_NN/trajectory.json`` files hold only what the engine reports (for
Codex, the agent's steps). ``trajectory_full.json`` puts the whole run in one place, in
order: the system instructions, developer message and environment context the model was
given, every user turn (including follow-ups in later sessions), and every agent step.
System and user steps are copied verbatim from each session's first captured API request
(``raw_dumps/request_001.json``), so they show exactly what the model saw; the request's
tool list becomes ``agent.tool_definitions``. Without captured requests, user turns come
from the session trajectories or, failing that, the run's config.yaml.

Steps are renumbered 1..N across sessions; each keeps ``extra.session_index`` and, for
agent steps, ``extra.session_step_id`` (its number in the session's own trajectory).

Usage: python -m harness.full_trajectory RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import yaml

FILENAME = "trajectory_full.json"
_SESSION_DIR = re.compile(r"^session_(\d+)(?:_r(\d+))?$")
_REQUEST = re.compile(r"^request_(\d+)\.json$")


def _text(content: Any) -> str:
    """Flatten a message's content (string or list of blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                else:
                    parts.append(f"[{block.get('type', 'non-text')} content]")
        return "\n".join(parts)
    return str(content)


def _messages(req: dict) -> list[tuple[str, str]]:
    """(role, text) for a request's context, oldest first. Role is "system" for system and
    developer instructions, "user" for user messages, and "other" for anything the model
    produced or received as a tool result (assistant messages, calls, call outputs)."""
    out: list[tuple[str, str]] = []
    if isinstance(req.get("instructions"), str) and req["instructions"].strip():  # OpenAI Responses
        out.append(("system", req["instructions"]))
    if req.get("system"):  # Anthropic Messages
        out.append(("system", _text(req["system"])))
    for item in req.get("input") or req.get("messages") or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if item.get("type") not in (None, "message") or role not in ("system", "developer", "user"):
            out.append(("other", ""))
            continue
        if role == "user" and isinstance(item.get("content"), list) and item["content"] and all(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in item["content"]):
            out.append(("other", ""))  # Anthropic tool results arrive as user messages
            continue
        out.append(("system" if role in ("system", "developer") else "user", _text(item.get("content"))))
    return out


def _first_request(session_dir: Path) -> tuple[dict | None, Path | None]:
    dumps = session_dir / "raw_dumps"
    if not dumps.is_dir():
        return None, None
    reqs = sorted((int(m.group(1)), p) for p in dumps.iterdir() if (m := _REQUEST.match(p.name)))
    for _, p in reqs:
        try:
            return json.loads(p.read_text()), p
        except (OSError, json.JSONDecodeError):
            continue
    return None, None


def _session_dirs(run_dir: Path) -> list[tuple[int, int, Path]]:
    found = []
    for p in run_dir.iterdir():
        m = _SESSION_DIR.match(p.name)
        if m and p.is_dir() and (p / "trajectory.json").exists():
            found.append((int(m.group(1)), int(m.group(2) or 0), p))
    return sorted(found)


def _config_prompts(run_dir: Path) -> dict[int, str]:
    try:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return {int(s.get("session_index") or 1): str(s["prompt"]) for s in cfg.get("sessions") or []
            if isinstance(s, dict) and s.get("prompt")}


def build_full_trajectory(run_dir: str | Path, session_mode: str | None = None) -> dict:
    """Build the run-level ATIF trajectory (a JSON-ready dict) for ``run_dir``."""
    run_dir = Path(run_dir)
    if session_mode is None:
        try:
            session_mode = json.loads((run_dir / "run_meta.json").read_text()).get("session_mode")
        except (OSError, json.JSONDecodeError):
            session_mode = None
    prompts = _config_prompts(run_dir)
    sessions = _session_dirs(run_dir)
    if not sessions:
        raise FileNotFoundError(f"no session_NN/trajectory.json under {run_dir}")

    steps: list[dict] = []
    agent: dict[str, Any] | None = None
    tool_definitions: list | None = None
    session_info: list[dict] = []
    first_session_id = None

    def add(step: dict) -> None:
        step["step_id"] = len(steps) + 1
        steps.append(step)

    for n, (index, rep, sdir) in enumerate(sessions):
        traj = json.loads((sdir / "trajectory.json").read_text())
        own = traj.get("steps") or []
        agent = agent or traj.get("agent")
        first_session_id = first_session_id or traj.get("session_id")
        ts = own[0].get("timestamp") if own else None
        req, req_path = _first_request(sdir)
        origin = str(req_path.relative_to(run_dir)) if req_path else None
        base = {"session_index": index, **({"replicate": rep} if rep else {})}
        # A later session continues the conversation (chained, forked, branch prefix) unless
        # the run is isolated: then each session is a new conversation with its own preamble.
        new_conversation = n == 0 or session_mode == "isolated"

        if req is not None:
            if tool_definitions is None and isinstance(req.get("tools"), list):
                tool_definitions = req["tools"]
            msgs = _messages(req)
            if new_conversation:
                # everything before the model's first output in this conversation
                cut = next((i for i, (r, _) in enumerate(msgs) if r == "other"), len(msgs))
                context = msgs[:cut]
            else:
                # the new turn: trailing system/user items after the carried-over history
                cut = len(msgs)
                while cut > 0 and msgs[cut - 1][0] != "other":
                    cut -= 1
                context = msgs[cut:]
            for role, text in context:
                add({"step_id": 0, "timestamp": ts, "source": role, "message": text,
                     "extra": {**base, "origin": origin}})
            agent_only = [s for s in own if s.get("source") == "agent"]
        else:
            agent_only = own
            if not any(s.get("source") == "user" for s in own) and index in prompts:
                add({"step_id": 0, "timestamp": ts, "source": "user", "message": prompts[index],
                     "extra": {**base, "origin": "config.yaml"}})

        for s in agent_only:
            step = json.loads(json.dumps(s))
            step["extra"] = {**(step.get("extra") or {}), **base, "session_step_id": s.get("step_id")}
            add(step)
        session_info.append({**base, "dir": sdir.name, "steps": len(own), "first_request": origin,
                             "new_conversation": new_conversation})

    agent = dict(agent or {"name": "unknown", "version": "unknown"})
    if tool_definitions:
        agent["tool_definitions"] = tool_definitions
    full_diff = run_dir / "full_diff.patch"
    return {
        "schema_version": "ATIF-v1.6",
        "session_id": f"{first_session_id or run_dir.name}__full",
        "agent": agent,
        "steps": steps,
        "notes": ("Run-level view for reading and analysis: system context, every user turn and "
                  "agent step, in order. Not used for judging (judges render session_01 only)."),
        "extra": {"run_dir": run_dir.name, "session_mode": session_mode, "sessions": session_info,
                  "full_diff": full_diff.read_text() if full_diff.exists() else None},
    }


def write_full_trajectory(run_dir: str | Path, session_mode: str | None = None) -> Path:
    """Build, validate against the ATIF schema, and write ``trajectory_full.json``."""
    from harbor.models.trajectories.trajectory import Trajectory

    run_dir = Path(run_dir)
    data = build_full_trajectory(run_dir, session_mode)
    Trajectory.model_validate(data)
    out = run_dir / FILENAME
    out.write_text(json.dumps(data, indent=2, default=str))
    return out


def write_full_trajectory_safely(run_dir: str | Path, session_mode: str | None = None) -> Path | None:
    """write_full_trajectory for the end of a run: a failure is logged, never raised."""
    try:
        return write_full_trajectory(run_dir, session_mode)
    except Exception as e:
        logging.getLogger(__name__).warning("trajectory_full.json not written for %s: %r", run_dir, e)
        return None


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(write_full_trajectory(arg))
