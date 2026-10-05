"""Branch rollouts (pipeline/branch.py, harness/prefill.py, harness/resume.py)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from harness.prefill import (
    BranchUnsupportedError,
    check_branch_support,
    parse_harmony,
    parse_inkling,
    parse_kimi,
    responses_sse,
    ParsedOutput,
)
from harness.reasoning_capture import parse_responses_sse
from harness.resume import RESUME_MARKER, strip_resume_additions
from harness.runner import _is_startup_notice
from harness.shadow_git import ShadowGit

# --------------------------------------------------------------------------- parsers

def test_parse_inkling_stripped_tokens():
    p = parse_inkling('so I will list.exec_command{"name":"exec_command","args":{"cmd":"ls -la"}}')
    assert p.complete and p.reasoning == "so I will list." and p.tool_calls[0]["name"] == "exec_command"
    assert json.loads(p.tool_calls[0]["arguments"]) == {"cmd": "ls -la"}
    assert not parse_inkling("just thinking, no action").complete


def test_parse_kimi_with_spaced_markers():
    text = (' more thought. </think> I will write it. <|tool_calls_section_begin|> <|tool_call_begin|> '
            'functions.exec_command:3 <|tool_call_argument_begin|>{"cmd": "cat > cache.py"}<|tool_call_end|> '
            '<|tool_calls_section_end|>')
    p = parse_kimi(text)
    assert p.complete and p.reasoning == " more thought. " and p.message == "I will write it."
    assert p.tool_calls == [{"name": "exec_command", "arguments": '{"cmd": "cat > cache.py"}'}]
    assert not parse_kimi("still thinking").complete


def test_parse_harmony():
    text = ('so run ls.<|end|><|start|>assistant<|channel|>commentary to=functions.exec_command '
            '<|constrain|>json<|message|>{"cmd":"ls"}<|call|>')
    p = parse_harmony(text)
    assert p.complete and p.reasoning == "so run ls." and p.tool_calls[0] == {"name": "exec_command", "arguments": '{"cmd":"ls"}'}
    p = parse_harmony("done.<|end|><|start|>assistant<|channel|>final<|message|>All good.<|return|>")
    assert p.message == "All good." and not p.tool_calls


def test_responses_sse_round_trips_through_capture():
    sse = responses_sse("m", "PREFIX + continuation", ParsedOutput("x", [{"name": "exec_command", "arguments": '{"cmd": "ls -la"}'}]), {})
    rec = parse_responses_sse(_write(Path(__import__("tempfile").mkdtemp()) / "r.txt", sse.decode()), 1)
    assert rec.text == "PREFIX + continuation" and rec.n_actions == 1 and rec.kind == "raw"


def _write(p: Path, text: str) -> Path:
    p.write_text(text)
    return p


# --------------------------------------------------------------------------- support matrix

@pytest.mark.parametrize("engine,provider,model,order,match", [
    ("claude_code", "openrouter", "thinkingmachines/inkling", ["together"], "Claude Code"),
    ("codex", "openai", "gpt-5.5", None, "openrouter"),
    ("codex", "openrouter", "google/gemini-3.1-pro-preview", ["google-vertex"], "Closed models"),
    ("codex", "openrouter", "thinkingmachines/inkling", None, "branch.provider"),
    ("codex", "openrouter", "thinkingmachines/inkling", ["deepinfra/fp8"], "not verified"),
])
def test_unsupported_combinations_raise_clearly(engine, provider, model, order, match):
    with pytest.raises(BranchUnsupportedError, match=match):
        check_branch_support(engine, provider, model, order)


def test_supported_combination():
    assert check_branch_support("codex", "openrouter", "thinkingmachines/inkling", ["together"]).providers["together"] == "verified"
    assert check_branch_support("codex", "openrouter", "moonshotai/kimi-k2.6", ["crusoe/bf16"]).providers["crusoe/bf16"] == "verified"
    assert check_branch_support("codex", "openrouter", "openai/gpt-oss-120b", ["cerebras/fp16"]).providers["cerebras/fp16"] == "experimental"
    with pytest.raises(BranchUnsupportedError, match="not verified"):
        check_branch_support("codex", "openrouter", "openai/gpt-oss-120b", ["deepinfra/bf16"])


# --------------------------------------------------------------------------- resume stripping

def _m(role, text):
    return {"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]}


def test_strip_resume_additions():
    history = [_m("developer", "perms"), _m("user", "<environment_context>a</environment_context>"), _m("user", "task")]
    resumed = history + [_m("developer", "perms2"), _m("user", "<environment_context>b</environment_context>"),
                         _m("user", RESUME_MARKER)]
    later = resumed + [{"type": "function_call", "name": "x"}]
    assert strip_resume_additions(resumed, keep=3) == history
    assert strip_resume_additions(later, keep=3) == history + [{"type": "function_call", "name": "x"}]
    assert strip_resume_additions(history, keep=3) == history          # no marker: untouched


def test_startup_notice_detection():
    assert _is_startup_notice({"source": "agent", "tool_calls": [{"function_name": "error"}]})
    assert not _is_startup_notice({"source": "agent", "tool_calls": [{"function_name": "exec_command"}]})


# --------------------------------------------------------------------------- config validation

def _sweep(**kw):
    from tests.test_isolation_realism import _sweep as base
    return base(**kw)


def test_branch_sweep_validation():
    arms = {"cut": {"prefix_until": "Actually"}, "keep": {"full_original": True}}
    br = {"seed_run": "x", "request": 4, "arms": arms}
    cfg = _sweep(pressure={"variable": "branch_arm", "values": ["cut", "keep"]}, branch=br)
    assert cfg.branch.request == 4
    with pytest.raises(ValueError, match="needs a branch"):
        _sweep(pressure={"variable": "branch_arm", "values": ["cut"]})
    with pytest.raises(ValueError, match="requires pressure.variable"):
        _sweep(branch=br)
    with pytest.raises(ValueError, match="no arm named"):
        _sweep(pressure={"variable": "branch_arm", "values": ["nope"]}, branch=br)
    with pytest.raises(ValueError, match="exactly one"):
        _sweep(pressure={"variable": "branch_arm", "values": ["a"]},
               branch={**br, "arms": {"a": {"prefix_until": "x", "full_original": True}}})


# --------------------------------------------------------------------------- seed + splice

REASONING = "Look at the code. Actually, let me think about whether I can cheat. No, do it properly."


def _sse_items(items, input_tokens=100):
    evs = [{"type": "response.output_item.done", "output_index": i, "item": it} for i, it in enumerate(items)]
    evs.append({"type": "response.completed", "response": {"usage": {"input_tokens": input_tokens}}})
    return "".join(f"data: {json.dumps(e)}\n\n" for e in evs)


@pytest.fixture
def seed_run(tmp_path):
    """A minimal seed run: 2 requests; request 2 is the branch point (step 3)."""
    run = tmp_path / "seed"
    s = run / "session_01"
    (s / "raw_dumps").mkdir(parents=True)
    old_cwd = "/tmp/ws/aaaaaaaaaaaa"
    work = tmp_path / "work"
    work.mkdir()
    (work / "cache.py").write_text("x = 1\n")
    g = ShadowGit(work_dir=work, git_dir=run / ".shadow_git")
    g.init()
    g.commit_baseline()
    (work / "cache.py").write_text("x = 2\n")
    g._git("add", "-A")
    g._git("commit", "-q", "-m", "s2")
    g.tag("_step_1_2")
    hist = [_m("developer", "perms"), _m("user", f"<environment_context><cwd>{old_cwd}</cwd></environment_context>"), _m("user", "do the task")]
    r1_items = [{"type": "reasoning", "content": [{"type": "reasoning_text", "text": "first look"}]},
                {"type": "function_call", "name": "exec_command", "call_id": "c1", "arguments": json.dumps({"cmd": "ls -la"})}]
    req1 = {"model": "thinkingmachines/inkling", "input": hist, "tools": [{"type": "function", "name": "exec_command"}]}
    req2 = {**req1, "input": hist + r1_items + [{"type": "function_call_output", "call_id": "c1", "output": "cache.py"}]}
    r2_items = [{"type": "reasoning", "content": [{"type": "reasoning_text", "text": REASONING}]},
                {"type": "function_call", "name": "exec_command", "call_id": "c2", "arguments": json.dumps({"cmd": "python3 t.py"})}]
    for i, (req, items) in enumerate([(req1, r1_items), (req2, r2_items)], 1):
        (s / "raw_dumps" / f"request_{i:03d}.json").write_text(json.dumps(req))
        (s / "raw_dumps" / f"request_{i:03d}_headers.json").write_text(json.dumps({"path": "/v1/responses"}))
        (s / "raw_dumps" / f"response_{i:03d}.txt").write_text(_sse_items(items))
    steps = [{"step_id": 1, "source": "agent", "tool_calls": [{"function_name": "error", "arguments": {"message": "w"}}]},
             {"step_id": 2, "source": "agent", "tool_calls": [{"function_name": "command_execution", "arguments": {"command": "/bin/bash -lc 'ls -la'"}}]},
             {"step_id": 3, "source": "agent", "tool_calls": [{"function_name": "command_execution", "arguments": {"command": "/bin/bash -lc 'python3 t.py'"}}]}]
    (s / "trajectory.json").write_text(json.dumps({"steps": steps}))
    lines = [{"type": "session_meta", "payload": {"id": "old", "session_id": "old", "cwd": old_cwd}},
             {"type": "turn_context", "payload": {"cwd": old_cwd}}]
    lines += [{"type": "response_item", "payload": it} for it in req2["input"] + r2_items]
    (s / "transcript.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    cfg = {"engine": "codex", "model": "thinkingmachines/inkling", "provider": "openrouter", "provider_order": ["together"],
           "work_dir": str(work), "sessions": [{"session_index": 1, "prompt": "do the task"}]}
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    return run, old_cwd


def test_seed_locates_branch_step_and_arms(seed_run):
    from pipeline.branch import Seed
    run, old_cwd = seed_run
    seed = Seed(run, 2)
    assert seed.branch_step_id == 3 and seed.reset_tag == "_step_1_2" and seed.provider == "together"
    assert [s["step_id"] for s in seed.prefix_steps] == [1, 2]
    assert seed.prefix_steps[1]["reasoning_content"] == "first look"      # recovered for the judge
    assert seed.prefix_for({"prefix_until": "Actually,"}) == "Look at the code."   # trailing space stripped
    assert seed.prefix_for({"prefix_until": "Actually,", "append": " Wait.\n\n"}) == "Look at the code.  Wait."
    assert seed.prefix_for({"prefix_through": "cheat."}) == "Look at the code. Actually, let me think about whether I can cheat."
    assert seed.prefix_for({"full_original": True, "append": " Hmm."}) == REASONING + " Hmm."
    with pytest.raises(ValueError, match="not found"):
        seed.prefix_for({"prefix_until": "nonexistent"})
    entries, sid = seed.truncated_rollout("/tmp/ws/bbbbbbbbbbbb")
    assert sum(e["type"] == "response_item" for e in entries) == len(seed.request["input"])
    assert old_cwd not in json.dumps(entries) and entries[0]["payload"]["id"] == sid


def test_splice_answers_first_request_and_forwards_later(seed_run, monkeypatch):
    from pipeline import branch
    run, old_cwd = seed_run
    seed = branch.Seed(run, 2)
    new_cwd = "/tmp/ws/bbbbbbbbbbbb"
    calls = {}

    def fake_render(req, spec, prefix, drop_past_reasoning=False, provider=None):
        calls["prefix"] = prefix
        return "PROMPT", 42

    async def fake_complete(model, provider, prompt, api_key, **kw):
        calls["provider"] = provider
        return {"choices": [{"text": ' carefully.exec_command{"name":"exec_command","args":{"cmd":"pytest"}}'}],
                "usage": {"prompt_tokens": 42, "completion_tokens": 9}}

    monkeypatch.setattr(branch, "render_prompt", fake_render)
    monkeypatch.setattr(branch, "complete_raw", fake_complete)
    prefix = seed.prefix_for({"prefix_until": "Actually,"})
    splice = branch.BranchSplice(seed, prefix, new_cwd, "key")
    expected = json.loads(json.dumps(seed.request["input"]).replace(old_cwd, new_cwd))
    req = {"input": expected + [_m("developer", "p"), _m("user", "<environment_context>x</environment_context>"), _m("user", RESUME_MARKER)]}
    sse = asyncio.run(splice(req, 1))
    assert req["input"] == expected and splice.meta["first_request_matches_seed"] is True
    assert splice.meta["action_parsed"] and calls == {"prefix": prefix, "provider": "together"}
    rec = parse_responses_sse(_write(run / "x.txt", sse.decode()), 1)
    assert rec.text == prefix + " carefully." and rec.n_actions == 1
    assert splice.meta["n_attempts"] == 1
    later = {"input": req["input"] + [_m("developer", "p"), _m("user", RESUME_MARKER), {"type": "function_call_output", "output": "ok"}]}
    assert asyncio.run(splice(later, 2)) is None and RESUME_MARKER not in json.dumps(later)


def _splice_with(seed_run, monkeypatch, texts, **req_extra):
    from pipeline import branch
    run, old_cwd = seed_run
    seed = branch.Seed(run, 2)
    seen = []

    async def fake_complete(model, provider, prompt, api_key, **kw):
        seen.append(kw)
        return {"choices": [{"text": texts[min(len(seen), len(texts)) - 1], "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 42}}

    monkeypatch.setattr(branch, "render_prompt", lambda *a, **k: ("PROMPT", 42))
    monkeypatch.setattr(branch, "complete_raw", fake_complete)
    splice = branch.BranchSplice(seed, "Look.", "/tmp/ws/bbbbbbbbbbbb", "key")
    req = {"input": json.loads(json.dumps(seed.request["input"]).replace(old_cwd, "/tmp/ws/bbbbbbbbbbbb")), **req_extra}
    asyncio.run(splice(req, 1))
    return splice, seen


GOOD = ' ok.exec_command{"name":"exec_command","args":{"cmd":"pytest"}}'


def test_splice_resamples_unparsed_action(seed_run, monkeypatch):
    splice, seen = _splice_with(seed_run, monkeypatch, [" I will run", GOOD])
    assert splice.meta["n_attempts"] == 2 and splice.meta["action_parsed"] is True
    assert [a["action_parsed"] for a in splice.meta["attempts"]] == [False, True]
    assert splice.meta["tool_calls"][0]["name"] == "exec_command"


def test_splice_gives_up_after_max_attempts(seed_run, monkeypatch):
    splice, seen = _splice_with(seed_run, monkeypatch, [" no action"])
    assert splice.meta["n_attempts"] == 3 and splice.meta["action_parsed"] is False and len(seen) == 3


def test_splice_uses_request_sampling(seed_run, monkeypatch):
    splice, seen = _splice_with(seed_run, monkeypatch, [GOOD], temperature=0.7, top_p=0.9,
                                reasoning={"effort": "high"})
    assert seen == [{"temperature": 0.7, "top_p": 0.9}]
    assert splice.meta["sampling"] == {"reasoning_effort": "high", "temperature": 0.7, "top_p": 0.9}
    assert splice.meta["sampling_matches_seed"] is False   # the fixture seed sent none


def test_seed_fidelity_check(seed_run, monkeypatch):
    from pipeline import branch
    run, _ = seed_run
    seed = branch.Seed(run, 2)
    assert seed.provider_input_tokens == 100
    monkeypatch.setattr(branch, "render_chat", lambda *a, **k: "X")
    monkeypatch.setattr(branch, "count_tokens", lambda spec, text: 100)
    assert seed.check_fidelity(0) == {"rendered_tokens": 100, "provider_tokens": 100, "gap": 0}
    seed._fidelity = None
    monkeypatch.setattr(branch, "count_tokens", lambda spec, text: 96)
    with pytest.raises(BranchUnsupportedError, match="gap -4"):
        seed.check_fidelity(0)
    assert seed.check_fidelity(4)["gap"] == -4


def test_full_diff_is_against_seed_baseline(seed_run, tmp_path):
    from pipeline.branch import _full_diff
    run, _ = seed_run
    final = tmp_path / "final"
    final.mkdir()
    (final / "cache.py").write_text("x = 3\n")
    (final / "new.py").write_text("y\n")
    d = _full_diff(run / ".shadow_git", final)
    assert "-x = 1" in d and "+x = 3" in d and "new.py" in d


def test_codex_resume_argv():
    from harness.engines.base import EngineRunSpec
    from harness.engines.codex import CodexEngine
    argv = CodexEngine()._build_argv(EngineRunSpec(prompt="p", model="thinkingmachines/inkling", cwd="/w",
                                                   provider="openrouter", sandbox_mode="danger-full-access",
                                                   resume_session_id="sid"), "-")
    assert argv[:4] == ["codex", "exec", "resume", "sid"] and "-C" not in argv and "-s" not in argv
    assert 'sandbox_mode="danger-full-access"' in argv and argv[-1] == "-"
    with pytest.raises(ValueError, match="experimental_resume"):
        CodexEngine()._build_argv(EngineRunSpec(prompt="p", model="m", cwd="/w", resume_rollout_path="/x"), "p")
