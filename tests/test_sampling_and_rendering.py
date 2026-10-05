"""Sampling (reasoning effort / temperature / top_p), Codex model limits, and the
per-provider prompt rendering used by branch prefills."""

from __future__ import annotations

import json

import pytest

from harness import model_limits
from harness.config import RunConfig
from harness.engines.base import EngineRunSpec
from harness.engines.codex import CodexEngine
from harness.prefill import (
    NS_SEP,
    SUPPORTED,
    ParsedOutput,
    RenderStyle,
    recommended_sampling,
    request_to_messages,
    responses_sse,
    tools_to_chat,
)
from harness.proxy import apply_sampling


def _rc(**kw) -> RunConfig:
    base = dict(engine="codex", model="moonshotai/kimi-k2.6", provider="openrouter", work_dir="/w",
                sessions=[{"session_index": 1, "prompt": "p"}])
    base.update(kw)
    return RunConfig(**base)


def _m(role, *parts):
    return {"type": "message", "role": role, "content": [{"type": "input_text", "text": t} for t in parts]}


# --- sampling ---------------------------------------------------------------

def test_recommended_sampling_table():
    assert recommended_sampling("moonshotai/kimi-k2.6") == (1.0, 0.95)
    assert recommended_sampling("thinkingmachines/inkling") == (1.0, 1.0)
    assert recommended_sampling("z-ai/glm-5.3") == (None, None)


def test_resolve_sampling_defaults_and_overrides():
    s = model_limits.resolve_sampling(_rc())
    assert s == {"reasoning_effort": "high", "temperature": 1.0, "top_p": 0.95,
                 "source": "config + model recommendation"}
    s = model_limits.resolve_sampling(_rc(temperature=0.6, reasoning_effort="low"))
    assert (s["temperature"], s["top_p"], s["reasoning_effort"]) == (0.6, 0.95, "low")
    # an explicit Codex override of the effort wins
    s = model_limits.resolve_sampling(_rc(codex_config_overrides=['model_reasoning_effort="medium"']))
    assert s["reasoning_effort"] == "medium"
    # branch rollouts: exactly what was given, None = not sent
    s = model_limits.resolve_sampling(_rc(use_recommended_sampling=False, reasoning_effort=None))
    assert s == {"reasoning_effort": None, "temperature": None, "top_p": None, "source": "config"}
    # unknown model: nothing recommended
    assert model_limits.resolve_sampling(_rc(model="z-ai/glm-5.3"))["temperature"] is None
    # no capture proxy -> temperature/top_p cannot be applied
    s = model_limits.resolve_sampling(_rc(capture_api_requests=False))
    assert s["temperature"] is None and s["reasoning_effort"] == "high"
    assert model_limits.resolve_sampling(_rc(engine="claude_code", provider="anthropic", model="claude")) == {}


def test_sampling_config_validation():
    with pytest.raises(ValueError, match="engine: codex only"):
        _rc(engine="claude_code", provider="anthropic", model="claude", temperature=0.5)
    with pytest.raises(ValueError, match="capture_api_requests"):
        _rc(capture_api_requests=False, top_p=0.9)


def test_proxy_apply_sampling():
    req = {"reasoning": {"summary": "auto"}, "temperature": 0.2}
    apply_sampling(req, {"temperature": 1.0, "top_p": 0.95, "reasoning_effort": "high"})
    assert req == {"reasoning": {"summary": "auto", "effort": "high"}, "temperature": 1.0, "top_p": 0.95}
    req = {"reasoning": {"effort": "low"}}
    apply_sampling(req, {"reasoning_effort": "high"})
    assert req == {"reasoning": {"effort": "low"}}            # the client's own effort is kept
    req = {"reasoning": None}
    apply_sampling(req, {"reasoning_effort": "high"})
    assert req == {"reasoning": {"effort": "high"}}


# --- Codex model limits -----------------------------------------------------

ENDPOINTS = [{"tag": "crusoe/bf16", "context_length": 262144, "max_completion_tokens": 65536},
             {"tag": "parasail/int4", "context_length": 131072, "max_completion_tokens": None},
             {"tag": "together", "context_length": 200000, "max_completion_tokens": 32000}]


def test_pick_limits():
    assert model_limits._pick(ENDPOINTS, ["crusoe/bf16"]) == {
        "context_window": 262144, "max_output_tokens": 65536, "endpoint": "crusoe/bf16"}
    assert model_limits._pick(ENDPOINTS, ["parasail"])["context_window"] == 131072   # bare provider name
    assert model_limits._pick(ENDPOINTS, ["nobody"]) is None
    unpinned = model_limits._pick(ENDPOINTS, None)
    assert (unpinned["context_window"], unpinned["max_output_tokens"]) == (131072, 32000)


def test_resolve_codex_limits(monkeypatch):
    monkeypatch.setattr(model_limits, "openrouter_limits",
                        lambda model, order: {"context_window": 1000, "max_output_tokens": 50, "endpoint": "x"})
    assert model_limits.resolve_codex_limits(_rc())["context_window"] == 1000
    lim = model_limits.resolve_codex_limits(_rc(codex_config_overrides=["model_context_window=7"]))
    assert lim["context_window"] is None and lim["max_output_tokens"] == 50
    assert model_limits.resolve_codex_limits(_rc(codex_model_limits=False)) is None
    assert model_limits.resolve_codex_limits(_rc(provider="openai", model="gpt-5")) is None


def _argv(**extra):
    spec = EngineRunSpec(prompt="p", model="moonshotai/kimi-k2.6", cwd="/w", provider="openrouter", extra=extra)
    return CodexEngine()._build_argv(spec, "p")


def test_codex_argv_effort_and_limits():
    argv = _argv(codex_reasoning_effort="high",
                 codex_model_limits={"context_window": 262144, "max_output_tokens": 65536})
    assert 'model_reasoning_effort="high"' in argv
    assert "model_context_window=262144" in argv and "model_max_output_tokens=65536" in argv
    argv = _argv(codex_reasoning_effort="high", codex_model_limits={"context_window": 262144},
                 codex_config_overrides=["model_reasoning_effort=low", "model_context_window=9"])
    assert 'model_reasoning_effort="high"' not in argv and "model_reasoning_effort=low" in argv
    assert "model_context_window=262144" not in argv and "model_context_window=9" in argv
    argv = _argv()
    assert not any("model_reasoning_effort" in a or "model_context_window" in a for a in argv)


# --- provider rendering styles ----------------------------------------------

REQ = {"instructions": "INSTR",
       "input": [_m("developer", "P1", "P2"), _m("developer", "D2"), _m("user", "U1", "U2")]}


def test_render_style_default():
    msgs = request_to_messages(REQ)
    assert msgs == [{"role": "system", "content": "INSTR"}, {"role": "system", "content": "P1P2"},
                    {"role": "system", "content": "D2"}, {"role": "user", "content": "U1U2"}]


def test_render_style_merge_and_part_sep():
    msgs = request_to_messages(REQ, style=RenderStyle(part_sep=" ", merge_system="\n\n"))
    assert msgs == [{"role": "system", "content": "INSTR\n\nP1 P2\n\nD2"}, {"role": "user", "content": "U1 U2"}]


def test_render_style_system_into_first():
    req = {"instructions": "INSTR", "input": [_m("user", "u"), _m("developer", "LATE")]}
    msgs = request_to_messages(req, style=RenderStyle(system_into_first=True, merge_system="\n\n"))
    assert msgs == [{"role": "system", "content": "INSTR\n\nLATE"}, {"role": "user", "content": "u"}]


def test_measured_provider_styles():
    assert SUPPORTED["thinkingmachines/inkling"].style_for("together") == RenderStyle(part_sep=" ", merge_system="\n\n")
    assert SUPPORTED["moonshotai/kimi-k2.6"].style_for("crusoe/bf16") == RenderStyle(part_sep="\n")
    assert SUPPORTED["moonshotai/kimi-k2.6"].style_for("parasail/int4") == RenderStyle()


def test_namespace_tools_flattened_and_mapped_back():
    tools = [{"type": "namespace", "name": "multi_agent_v1", "tools": [{"type": "function", "name": "spawn_agent"}]}]
    assert tools_to_chat(tools)[0]["function"]["name"] == f"multi_agent_v1{NS_SEP}spawn_agent" == "multi_agent_v1__spawn_agent"
    call = {"type": "function_call", "name": "spawn_agent", "namespace": "multi_agent_v1", "call_id": "c", "arguments": "{}"}
    msgs = request_to_messages({"input": [_m("user", "u"), call]})
    assert msgs[-1]["tool_calls"][0]["function"]["name"] == "multi_agent_v1__spawn_agent"
    sse = responses_sse("m", "r", ParsedOutput(reasoning="r", tool_calls=[
        {"name": "multi_agent_v1__spawn_agent", "arguments": "{}"}, {"name": "a__b", "arguments": "{}"}]),
        None, {"multi_agent_v1"}).decode()
    done = [json.loads(l[5:]) for l in sse.splitlines() if l.startswith("data:") and "response.completed" in l][0]
    calls = [o for o in done["response"]["output"] if o["type"] == "function_call"]
    assert (calls[0]["name"], calls[0]["namespace"]) == ("spawn_agent", "multi_agent_v1")
    assert calls[1]["name"] == "a__b" and "namespace" not in calls[1]   # not a known namespace


def test_run_meta_records_sampling(tmp_path):
    from harness.experiment import _build_run_meta
    from harness.runner import SessionResult
    from harness.shadow_git import ShadowGit
    from harness.state import StateManager

    (tmp_path / "w").mkdir()
    state = StateManager(work_dir=tmp_path / "w", shadow_git=ShadowGit(tmp_path / "w", tmp_path / "g"))
    s = {"reasoning_effort": "high", "temperature": 1.0, "top_p": 0.95, "source": "config"}
    lim = {"context_window": 262144, "max_output_tokens": 65536, "endpoint": "crusoe/bf16"}
    meta = _build_run_meta(_rc(), "r", [SessionResult(session_index=1, sampling=s, codex_model_limits=lim)], state)
    assert meta["sampling"] == s
    assert meta["sessions"][0]["sampling"] == s and meta["sessions"][0]["codex_model_limits"] == lim
