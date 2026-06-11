"""Render an AgentLens run into a single text document for the LLM judge.

Includes the task the agent was given (system + session prompts from the run's saved
config.yaml — NOT max_budget_usd, so the judge stays blind to the experimental
condition), the agent's reasoning + tool calls + (truncated) tool outputs, followed by
the run's full_diff.patch — which is the key evidence for file-level reward hacking
(edits to the scorer, training on the held-out set, etc.). Large pieces are
middle-elided and the whole thing is capped to a character budget, with the diff
preserved.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


# Files whose diffs are decisive evidence of reward hacking; kept (truncated if needed)
# even when large, before filling the rest of the budget with other files smallest-first.
# Semantic hints only (data tampering here means heldout/, already covered) to avoid
# over-prioritizing benign large data files.
_PRIORITY_HINTS = ("eval", "heldout", "scorer", "score", "reward", "test_", "/test")
_NOTE_RESERVE = 200  # leave room for the omitted-files marker so we don't exceed budget


def _section_name(section: str) -> str:
    m = re.search(r"^diff --git a/(\S+)", section, re.M)
    return m.group(1) if m else "?"


def _is_priority(section: str) -> bool:
    name = _section_name(section).lower()
    return any(h in name for h in _PRIORITY_HINTS)


def _annotate_section(section: str, step_map: dict[str, list[int]] | None) -> str:
    """Insert `[file modified at step(s): ...]` after a section's `diff --git` line,
    so the judge can ground first_step_id in the decisive file evidence."""
    if not step_map:
        return section
    steps = step_map.get(_section_name(section))
    if not steps:
        return section
    nl = section.find("\n")
    if nl < 0:
        return section
    note = f"\n[file modified at step(s): {', '.join(str(s) for s in steps)}]"
    return section[:nl] + note + section[nl:]


def _render_diff(diff: str, budget: int, step_map: dict[str, list[int]] | None = None) -> str:
    """Fit the diff into ~`budget` chars. Priority (suspicious) files are kept first —
    truncated rather than dropped if large — then the remaining budget is filled with
    other files smallest-first, so a decisive edit (e.g. to eval_heldout.py or the
    held-out data) survives even when big. Omitted files are named. The note is
    budget-reserved so the total stays within `budget`. Sections are annotated with
    the steps that modified the file (from the state changelog) when available."""
    sections = [s for s in re.split(r"(?m)(?=^diff --git )", diff) if s.strip()]
    sections = [_annotate_section(s, step_map) for s in sections]
    diff = "".join(sections) if sections else diff
    if len(diff) <= budget:
        return diff
    if len(sections) <= 1:
        return _trunc(diff, budget)

    avail = max(0, budget - _NOTE_RESERVE)
    kept: dict[int, str] = {}
    used = 0
    pri = [i for i in range(len(sections)) if _is_priority(sections[i])]
    other = [i for i in range(len(sections)) if i not in pri]

    for i in sorted(pri, key=lambda j: len(sections[j])):  # smallest priority first, then truncate big ones
        if used >= avail:
            break
        s = sections[i]
        if used + len(s) <= avail:
            kept[i] = s
            used += len(s)
        elif (avail - used) > 200:
            kept[i] = _trunc(s, avail - used)
            used = avail
    for i in sorted(other, key=lambda j: len(sections[j])):  # rest smallest-first, whole
        if used + len(sections[i]) <= avail:
            kept[i] = sections[i]
            used += len(sections[i])

    parts = [kept[i] for i in range(len(sections)) if i in kept]
    dropped = [i for i in range(len(sections)) if i not in kept]
    if dropped:
        names = [_section_name(sections[i]) for i in dropped]
        parts.append(f"\n...[{len(dropped)} file-diff(s) omitted to fit budget: "
                     f"{', '.join(names[:20])}]...\n")
    return "".join(parts)


def _trunc(text: str, limit: int) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    elided = len(text) - limit
    return f"{text[:head]}\n...[{elided} chars elided]...\n{text[-tail:]}"


def _format_args(args) -> str:
    try:
        s = json.dumps(args, ensure_ascii=False)
    except Exception:
        s = str(args)
    return _trunc(s, 600)


def _render_step_blocks(steps: list[dict], per_obs_limit: int = 2000) -> list[str]:
    """One rendered text block per step (labelled `[step N]` for judge citation)."""
    out: list[str] = []
    for st in steps:
        src = st.get("source")
        if src == "user":
            # The initial task prompt — include once for context.
            msg = st.get("message") or ""
            if msg.strip():
                out.append(f"[TASK PROMPT]\n{_trunc(msg, 1500)}\n")
            continue
        if src != "agent":
            continue
        sid = st.get("step_id")
        block = [f"[step {sid}]"]
        reasoning = st.get("reasoning_content")
        if reasoning:
            block.append(f"THINKING: {_trunc(reasoning, 2500)}")
        msg = st.get("message")
        if msg:
            block.append(f"SAID: {_trunc(msg, 2000)}")
        for tc in st.get("tool_calls") or []:
            name = tc.get("function_name", "?")
            block.append(f"TOOL CALL: {name}({_format_args(tc.get('arguments'))})")
        obs = st.get("observation") or {}
        for res in obs.get("results", []) or []:
            content = res.get("content", "")
            block.append(f"TOOL RESULT: {_trunc(content, per_obs_limit)}")
        out.append("\n".join(block))
    return out


_STEP_LABEL = re.compile(r"^\[step (\d+)\]")


def _fit_blocks(blocks: list[str], budget: int) -> str:
    """Fit whole step blocks into ~`budget` chars. Unlike character middle-elision,
    this drops WHOLE steps from the middle and leaves an explicit `[steps M-N
    omitted]` marker, so every `[step N]` label the judge sees refers to a step it
    can actually read (required for step-ID localization)."""
    joined = "\n\n".join(blocks)
    if len(joined) <= budget:
        return joined
    reserve = 120  # room for the omission marker
    head_budget = int((budget - reserve) * 0.6)
    tail_budget = (budget - reserve) - head_budget
    head: list[str] = []
    tail: list[str] = []
    i, j, used_h, used_t = 0, len(blocks) - 1, 0, 0
    while i <= j and used_h + len(blocks[i]) + 2 <= head_budget:
        head.append(blocks[i])
        used_h += len(blocks[i]) + 2
        i += 1
    while j >= i and used_t + len(blocks[j]) + 2 <= tail_budget:
        tail.insert(0, blocks[j])
        used_t += len(blocks[j]) + 2
        j -= 1
    if i > j:  # nothing dropped after all (rounding) — just join
        return "\n\n".join(head + tail)
    if not head and not tail:  # degenerate: one block larger than the whole budget
        return _trunc(joined, budget)
    ids = [m.group(1) for b in blocks[i:j + 1] if (m := _STEP_LABEL.match(b))]
    span = f"steps {ids[0]}-{ids[-1]}" if ids else f"{j - i + 1} blocks"
    marker = f"...[{span} ({j - i + 1} of {len(blocks)} steps) omitted to fit budget]..."
    return "\n\n".join(head + [marker] + tail)


def _load_task_description(run_dir: Path) -> str:
    """The instructions the agent was given, from the run's saved config.yaml.
    Only the prompts are included — the judge must stay blind to the experimental
    condition, so max_budget_usd and other run parameters are deliberately omitted."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return ""
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return ""
    parts: list[str] = []
    if cfg.get("system_prompt"):
        parts.append(f"[SYSTEM PROMPT]\n{str(cfg['system_prompt']).strip()}")
    for s in cfg.get("sessions") or []:
        if isinstance(s, dict) and s.get("prompt"):
            parts.append(f"[USER TASK PROMPT]\n{str(s['prompt']).strip()}")
    return "\n\n".join(parts)


def _write_steps_by_file(run_dir: Path) -> dict[str, list[int]]:
    """{file_path: sorted step_ids that modified it} from the state changelog."""
    path = run_dir / "state_changelog.jsonl"
    if not path.exists():
        return {}
    out: dict[str, list[int]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            w = json.loads(line)
        except Exception:
            continue
        fp, sid = w.get("file_path"), w.get("step_id")
        if fp and sid is not None and sid not in out.setdefault(str(fp), []):
            out[str(fp)].append(sid)
    return {k: sorted(v) for k, v in out.items()}


def render_trajectory(run_dir: str | Path, max_chars: int = 200_000) -> str:
    """Render run_dir (an AgentLens run directory) to a judge-input string."""
    run_dir = Path(run_dir)
    traj_path = run_dir / "session_01" / "trajectory.json"
    diff_path = run_dir / "full_diff.patch"

    try:
        traj = json.loads(traj_path.read_text())
        steps = traj.get("steps", [])
    except Exception as e:
        blocks = [f"(could not load trajectory: {e})"]
    else:
        blocks = _render_step_blocks(steps)

    diff = diff_path.read_text() if diff_path.exists() else "(no full_diff.patch found)"
    task = _trunc(_load_task_description(run_dir), 6_000) or "(task config not found)"

    # Reserve room for the task and the diff (key evidence), give the rest to the body.
    diff_budget = min(len(diff) + 2_000, max(20_000, int(max_chars * 0.4)))
    body_budget = max(5_000, max_chars - diff_budget - len(task) - 800)

    return (
        f"=== TASK GIVEN TO THE AGENT ===\n"
        f"{task}\n\n"
        f"=== AGENT TRAJECTORY ({run_dir.name}) ===\n"
        f"{_fit_blocks(blocks, body_budget)}\n\n"
        f"=== FINAL FILE DIFF (full_diff.patch) ===\n"
        f"{_render_diff(diff, diff_budget, _write_steps_by_file(run_dir))}\n"
    )
