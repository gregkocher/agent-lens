"""Reasoning is always recovered into trajectories, and judges see full trajectories."""

from __future__ import annotations

import json
from pathlib import Path

from harness.judge import render_trajectory_with_info
from harness.judge_budget import elide_middle, shrink_largest
from harness.reasoning_capture import (
    attach_reasoning,
    enrich_trajectory_reasoning,
    parse_anthropic_sse,
    records_from_codex_rollout,
    records_from_raw_dumps,
)

# --------------------------------------------------------------------------- fixtures

def _sse(items: list[dict]) -> str:
    evs = [{"type": "response.created", "response": {"output": []}}]
    evs += [{"type": "response.output_item.done", "output_index": i, "item": it} for i, it in enumerate(items)]
    evs.append({"type": "response.completed", "response": {"output": items, "usage": {}}})
    return "".join(f"data: {json.dumps(e)}\n\n" for e in evs)


def _reason(text=None, summary=None, encrypted=False):
    return {"type": "reasoning", "content": [{"type": "reasoning_text", "text": text}] if text else [],
            "summary": [{"type": "summary_text", "text": summary}] if summary else [],
            "encrypted_content": "gAAAA..." if encrypted else None}


def _call(cmd, **extra):
    return {"type": "function_call", "name": "exec_command", "call_id": "c",
            "arguments": json.dumps({"cmd": cmd, **extra})}


def _msg(text):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _write_dumps(session: Path, responses: list[list[dict]], contexts: dict[int, str] | None = None):
    raw = session / "raw_dumps"
    raw.mkdir(parents=True)
    caps = []
    for i, items in enumerate(responses, 1):
        (raw / f"request_{i:03d}.json").write_text(json.dumps({"input": []}))
        (raw / f"request_{i:03d}_headers.json").write_text(json.dumps({"path": "/v1/responses"}))
        (raw / f"response_{i:03d}.txt").write_text(_sse(items))
        caps.append({"request_index": i, "agent_context": (contexts or {}).get(i, "main")})
    (session / "api_captures.jsonl").write_text("".join(json.dumps(c) + "\n" for c in caps))


def _step(i, cmd=None, msg="", reasoning=None):
    st = {"step_id": i, "source": "agent", "message": msg}
    if cmd is not None:
        st["tool_calls"] = [{"function_name": "command_execution",
                             "arguments": {"command": f"/bin/bash -lc '{cmd}'"}}]
        st["observation"] = {"results": [{"content": "out"}]}
    if reasoning:
        st["reasoning_content"] = reasoning
    return st


# --------------------------------------------------------------------------- capture

def test_raw_summary_encrypted_kinds_and_subagents_skipped(tmp_path):
    _write_dumps(tmp_path, [
        [_reason(text="think A"), _call("ls -la")],
        [_reason(summary="summary B"), _call("cat a.py")],
        [_reason(encrypted=True), _call("python3 t.py")],
        [_reason(text="subagent thinking"), _call("echo sub")],
    ], contexts={4: "subagent"})
    recs = records_from_raw_dumps(tmp_path)
    assert [r.kind for r in recs] == ["raw", "summary", "encrypted"]
    steps = [_step(1, "ls -la"), _step(2, "cat a.py"), _step(3, "python3 t.py")]
    stats = attach_reasoning(steps, recs)
    assert steps[0]["reasoning_content"] == "think A" and steps[0]["extra"]["reasoning_kind"] == "raw"
    assert steps[1]["reasoning_content"] == "summary B" and steps[1]["extra"]["reasoning_kind"] == "summary"
    assert "reasoning_content" not in steps[2] and steps[2]["extra"]["reasoning_kind"] == "encrypted"
    assert stats["steps_filled"] == 3 and stats["unmatched_records"] == 0


def test_alignment_edge_cases_from_real_runs(tmp_path):
    wd = "/root/agent-lens/pipeline_runs/x/work_dirs/bp_r1"
    _write_dumps(tmp_path, [
        [_reason(text="R1"), _call(f"cd {wd} && cat cache_starter.py", workdir=wd)],   # shared workdir arg
        [_reason(text="R2"), _msg("Let me benchmark first."), _call("python3 test_cache.py", workdir=wd)],
        [_reason(text="R3"), _call("cat cache.py")],                                    # repeated command
        [_reason(text="R4"), _call("cat cache.py")],
        [_reason(text="R5 plan only"), {"type": "function_call", "name": "update_plan",
                                        "arguments": json.dumps({"plan": [{"step": "x", "status": "done"}]})}],
        [_reason(text="R6"), _call("ls")],                                              # short command
    ])
    steps = [
        {"step_id": 1, "source": "agent", "tool_calls": [{"function_name": "error", "arguments": {"message": "warn"}}]},
        _step(2, f"cd {wd} && cat cache_starter.py"),
        _step(3, msg="Let me benchmark first."),
        _step(4, "python3 test_cache.py"),
        _step(5, "cat cache.py"),
        _step(6, "cat cache.py"),
        _step(7, "ls"),
    ]
    stats = attach_reasoning(steps, records_from_raw_dumps(tmp_path))
    got = {s["step_id"]: s.get("reasoning_content") for s in steps}
    assert got[1] is None                       # engine warning pseudo-step
    assert got[2] == "R1" and got[3] == "R2" and got[4] is None   # 2nd action of the same response
    assert got[5] == "R3" and got[6] == "R4"     # repeated identical commands stay 1:1
    assert got[7] == "R5 plan only\n\nR6"        # step-less response rolls forward, nothing lost
    assert stats["unmatched_records"] == 0


def test_engine_reasoning_is_kept(tmp_path):
    _write_dumps(tmp_path, [[_reason(text="captured"), _call("ls -la")]])
    steps = [_step(1, "ls -la", reasoning="from the engine stream")]
    attach_reasoning(steps, records_from_raw_dumps(tmp_path))
    assert steps[0]["reasoning_content"] == "from the engine stream"


def test_codex_rollout_fallback(tmp_path):
    lines = [{"type": "session_meta", "payload": {}}]
    for item in [_reason(text="A"), _call("ls -la"), {"type": "function_call_output", "output": "x"},
                 _reason(text="B"), _call("cat a.py"), {"type": "function_call_output", "output": "y"}]:
        lines.append({"type": "response_item", "payload": item})
    (tmp_path / "transcript.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    traj = {"steps": [_step(1, "ls -la"), _step(2, "cat a.py")]}
    stats = enrich_trajectory_reasoning(traj, tmp_path, "codex")
    assert stats["source"] == "codex_rollout"
    assert [s["reasoning_content"] for s in traj["steps"]] == ["A", "B"]
    assert len(records_from_codex_rollout(tmp_path / "transcript.jsonl")) == 2


def test_summary_models_relabelled(tmp_path):
    _write_dumps(tmp_path, [[_reason(text="**Planning** I will list files"), _call("ls -la")]])
    traj = {"steps": [_step(1, "ls -la")]}
    enrich_trajectory_reasoning(traj, tmp_path, "codex", model="google/gemini-3.1-pro-preview")
    assert traj["steps"][0]["extra"]["reasoning_kind"] == "summary"
    traj = {"steps": [_step(1, "ls -la")]}
    enrich_trajectory_reasoning(traj, tmp_path, "codex", model="thinkingmachines/inkling")
    assert traj["steps"][0]["extra"]["reasoning_kind"] == "raw"


def test_anthropic_thinking_parsed(tmp_path):
    evs = [{"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
           {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me "}},
           {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "read it."}},
           {"type": "content_block_start", "index": 1, "content_block": {"type": "redacted_thinking"}},
           {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "name": "Read"}},
           {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"file_path": "/w/a.py"}'}}]
    p = tmp_path / "r.txt"
    p.write_text("".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in evs))
    r = parse_anthropic_sse(p, 1)
    assert r.reasoning == [("raw", "Let me read it."), ("encrypted", "")] and r.n_actions == 1


# --------------------------------------------------------------------------- judge budget

def test_shrink_largest_cuts_only_the_biggest():
    texts = ["a" * 100, "b" * 50_000, "c" * 20_000, "d" * 300]
    r = shrink_largest(texts, 30_000)
    assert r.texts[0] == texts[0] and r.texts[3] == texts[3]      # small outputs untouched
    assert len(r.texts[1]) < 50_000 and "characters omitted" in r.texts[1]
    assert sum(map(len, texts)) - sum(map(len, r.texts)) >= 30_000 and r.remaining_excess == 0
    assert elide_middle("x" * 10, 20) == "x" * 10


def test_live_judge_renders_full_and_shortens_largest_outputs_first():
    big = "L" * 60_000
    steps = [{"step_id": 1, "source": "agent", "reasoning_content": "R" * 5_000,
              "tool_calls": [{"function_name": "exec", "arguments": {"cmd": "x" * 2_000}}],
              "observation": {"results": [{"content": big}]}},
             {"step_id": 2, "source": "agent", "extra": {"reasoning_kind": "encrypted"},
              "message": "done", "observation": {"results": [{"content": "small"}]}}]
    full, info = render_trajectory_with_info(steps, max_chars=200_000)
    assert "R" * 5_000 in full and big in full and "x" * 2_000 in full and not info["truncated"]
    assert "encrypted" in full
    cut, info = render_trajectory_with_info(steps, max_chars=30_000)
    assert len(cut) <= 30_000 and info["truncated"] and info["tool_outputs_shortened"] == 1
    assert "R" * 5_000 in cut and "x" * 2_000 in cut and "[result] small" in cut   # never cut
    assert info["steps_omitted"] == 0


def test_pipeline_render_full_then_budgeted(tmp_path):
    from pipeline.render import render_info, render_trajectory
    s = tmp_path / "session_01"
    s.mkdir()
    steps = [{"step_id": 1, "source": "user", "message": "do the task"},
             {"step_id": 2, "source": "agent", "reasoning_content": "T" * 6_000, "extra": {"reasoning_kind": "raw"},
              "tool_calls": [{"function_name": "exec", "arguments": {"cmd": "y" * 3_000}}],
              "observation": {"results": [{"content": "O" * 80_000}]}},
             {"step_id": 3, "source": "agent", "extra": {"reasoning_kind": "summary"}, "reasoning_content": "S",
              "observation": {"results": [{"content": "tiny"}]}}]
    (s / "trajectory.json").write_text(json.dumps({"steps": steps}))
    (tmp_path / "full_diff.patch").write_text("diff --git a/cache.py b/cache.py\n+x\n")
    out = render_trajectory(tmp_path, 1_000_000)
    assert "T" * 6_000 in out and "O" * 80_000 in out and "y" * 3_000 in out     # nothing capped
    assert "THINKING (summary): S" in out and not render_info(tmp_path, 1_000_000)["truncated"]
    out = render_trajectory(tmp_path, 40_000)
    info = render_info(tmp_path, 40_000)
    assert len(out) <= 40_000 and info["truncated"] and info["tool_outputs_shortened"] == 1
    assert "T" * 6_000 in out and "TOOL RESULT: tiny" in out and "diff --git a/cache.py" in out
    assert info["steps_omitted"] == 0 and info["diff_chars_omitted"] == 0
