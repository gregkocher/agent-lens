"""Tests for the Codex engine's JSONL item -> EngineEvent translation.

These are pure translation tests — no subprocess, no network.
"""

from __future__ import annotations

from harness.atif_adapter import ATIFAdapter
from harness.engines.base import EngineRunSpec, classify_api_failure
from harness.engines.codex import (
    CodexEngine,
    classify_codex_nonzero_exit,
    codex_upstream,
)


def _engine():
    return CodexEngine()


def _spec(**overrides) -> EngineRunSpec:
    base = {"prompt": "hi", "model": "gpt-5-codex", "cwd": "/w"}
    base.update(overrides)
    return EngineRunSpec(**base)


class TestItemTranslation:
    def test_agent_message(self):
        ev = _engine()._translate_item({"id": "item_0", "type": "agent_message", "text": "pong"})
        assert ev.text == "pong"
        assert not ev.tool_calls

    def test_reasoning(self):
        ev = _engine()._translate_item({"id": "r0", "type": "reasoning", "text": "thinking"})
        assert ev.reasoning == "thinking"

    def test_command_execution_pairs_call_and_result(self):
        ev = _engine()._translate_item({
            "id": "item_3",
            "type": "command_execution",
            "command": "/bin/zsh -lc ls",
            "aggregated_output": "hello.txt\n",
            "exit_code": 0,
            "status": "completed",
        })
        assert ev.tool_calls[0].name == "command_execution"
        assert ev.tool_calls[0].id == "item_3"
        assert ev.tool_calls[0].arguments["command"] == "/bin/zsh -lc ls"
        assert ev.inline_results[0].tool_call_id == "item_3"
        assert "hello.txt" in ev.inline_results[0].content
        assert "exit_code=0" in ev.inline_results[0].content

    def test_file_change(self):
        ev = _engine()._translate_item({
            "id": "item_1",
            "type": "file_change",
            "changes": [{"path": "/tmp/codex_probe/hello.txt", "kind": "add"}],
            "status": "completed",
        })
        assert ev.tool_calls[0].name == "file_change"
        assert ev.inline_results[0].content.startswith("status=completed")
        assert "add /tmp/codex_probe/hello.txt" in ev.inline_results[0].content

    def test_web_search(self):
        ev = _engine()._translate_item({"id": "w0", "type": "web_search", "query": "foo"})
        assert ev.tool_calls[0].name == "web_search"
        assert ev.tool_calls[0].arguments["query"] == "foo"

    def test_unknown_item_preserved_as_tool_call(self):
        ev = _engine()._translate_item({"id": "x0", "type": "plan_update", "steps": ["a", "b"]})
        assert ev.tool_calls[0].name == "plan_update"
        assert ev.tool_calls[0].arguments["steps"] == ["a", "b"]


class TestSubagentSpawnTracking:
    def test_spawn_agent_recorded(self):
        eng = _engine()
        ev = eng._translate_item({
            "id": "item_2", "type": "collab_tool_call", "tool": "spawn_agent",
            "prompt": "read a.txt", "receiver_thread_ids": ["tid-1"],
            "agents_states": {"tid-1": {"status": "pending_init", "message": None}},
        })
        # Recorded for later linking
        assert len(eng._spawns) == 1
        assert eng._spawns[0]["thread_ids"] == ["tid-1"]
        assert eng._spawns[0]["item_id"] == "item_2"
        # Emitted as a spawn_agent tool call with an observation (ref anchor)
        assert ev.tool_calls[0].name == "spawn_agent"
        assert ev.inline_results[0].tool_call_id == "item_2"
        assert "tid-1" in ev.inline_results[0].content

    def test_wait_surfaces_child_result(self):
        eng = _engine()
        ev = eng._translate_item({
            "id": "item_5", "type": "collab_tool_call", "tool": "wait",
            "receiver_thread_ids": ["tid-1"],
            "agents_states": {"tid-1": {"status": "completed", "message": "- apple\n- banana"}},
        })
        assert ev.tool_calls[0].name == "wait"
        assert "apple" in ev.inline_results[0].content
        # wait does not register a new spawn
        assert eng._spawns == []


class TestUpstreamResolution:
    def test_default_is_openai(self):
        base, env_key, pid = codex_upstream(None, None)
        assert base == "https://api.openai.com/v1"
        assert env_key == "OPENAI_API_KEY"
        assert pid == "openai"

    def test_openrouter(self):
        base, env_key, pid = codex_upstream("openrouter", None)
        assert base == "https://openrouter.ai/api/v1"
        assert env_key == "OPENROUTER_API_KEY"
        assert pid == "openrouter"

    def test_base_url_override(self):
        base, _, _ = codex_upstream("openrouter", "https://example.test/v1")
        assert base == "https://example.test/v1"


class TestArgvConstruction:
    def test_openrouter_provider_block(self):
        argv = _engine()._build_argv(
            _spec(provider="openrouter", model="openai/gpt-5.3-codex"), "do it"
        )
        joined = " ".join(argv)
        assert 'model_provider="openrouter"' in joined
        assert 'model_providers.openrouter.base_url="https://openrouter.ai/api/v1"' in joined
        assert 'model_providers.openrouter.env_key="OPENROUTER_API_KEY"' in joined
        assert 'model_providers.openrouter.wire_api="responses"' in joined

    def test_openai_has_no_custom_provider_block(self):
        argv = _engine()._build_argv(_spec(provider="openai"), "do it")
        joined = " ".join(argv)
        assert "model_provider=" not in joined
        assert "model_providers" not in joined

    def test_capture_uses_openrouter_env_key(self):
        argv = _engine()._build_argv(
            _spec(
                provider="openrouter",
                model="openai/gpt-5.3-codex",
                capture_base_url="http://127.0.0.1:9999",
            ),
            "do it",
        )
        joined = " ".join(argv)
        # Capture routes through the proxy provider, not the openrouter block.
        assert 'model_provider="proxy"' in joined
        assert 'model_providers.proxy.env_key="OPENROUTER_API_KEY"' in joined
        assert 'model_providers.openrouter' not in joined


class TestUsageNormalization:
    def test_normalize_maps_keys(self):
        acc = {"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 80, "reasoning_output_tokens": 5}
        norm = CodexEngine._normalize_usage(acc)
        assert norm["input_tokens"] == 100
        assert norm["output_tokens"] == 20
        assert norm["cache_read_input_tokens"] == 80

    def test_accumulate_sums(self):
        acc: dict[str, int] = {}
        CodexEngine._accumulate_usage(acc, {"input_tokens": 10, "output_tokens": 2})
        CodexEngine._accumulate_usage(acc, {"input_tokens": 5, "output_tokens": 3})
        assert acc["input_tokens"] == 15
        assert acc["output_tokens"] == 5


class TestEndToEndTranslationToATIF:
    """Feed a realistic Codex item stream through the adapter."""

    def test_command_run_produces_paired_step(self):
        engine = _engine()
        adapter = ATIFAdapter("codex", "0.1.0", "gpt-5-codex", "session_01")

        events = [
            engine._translate_item({"id": "i0", "type": "agent_message", "text": "Creating file."}),
            engine._translate_item({
                "id": "i1", "type": "file_change",
                "changes": [{"path": "/w/hello.txt", "kind": "add"}], "status": "completed",
            }),
            engine._translate_item({
                "id": "i2", "type": "command_execution",
                "command": "ls", "aggregated_output": "hello.txt\n", "exit_code": 0, "status": "completed",
            }),
        ]
        for ev in events:
            adapter.process_event(ev)

        traj = adapter.build_trajectory()
        assert len(traj.steps) == 3
        # file_change step carries both tool call and observation
        fc = traj.steps[1]
        assert fc.tool_calls[0].function_name == "file_change"
        assert fc.observation.results[0].source_call_id == "i1"


class TestClassifyApiFailure:
    """Engine-agnostic error -> cause classification (rate_limited / auth_error / None)."""

    def test_429_is_rate_limited(self):
        assert classify_api_failure("stream error: HTTP 429 Too Many Requests") == "rate_limited"

    def test_rate_limit_phrase(self):
        assert classify_api_failure("Error: rate limit exceeded, retry") == "rate_limited"

    def test_overloaded_is_rate_limited(self):
        assert classify_api_failure("anthropic: Overloaded (529)") == "rate_limited"

    def test_5xx_and_timeout_are_rate_limited(self):
        assert classify_api_failure("503 Service Unavailable") == "rate_limited"
        assert classify_api_failure("upstream request timed out") == "rate_limited"

    def test_bad_key_is_auth_error(self):
        assert classify_api_failure(
            "Incorrect API key provided: sk-or-v1***a3c3") == "auth_error"

    def test_401_is_auth_error(self):
        assert classify_api_failure("HTTP 401 unauthorized") == "auth_error"

    def test_generic_error_is_none(self):
        assert classify_api_failure("Traceback ... KeyError: 'foo'") is None

    def test_empty_is_none(self):
        assert classify_api_failure("") is None
        assert classify_api_failure(None) is None

    def test_budget_value_not_a_false_positive(self):
        # A clean budget abort whose text mentions the limit must NOT look like an API
        # failure (else a real budget wall would be misclassified and re-run forever).
        assert classify_api_failure(
            "rollout budget of 200000 weighted tokens reached; exiting") is None


class TestCodexNonzeroExit:
    """The non-zero-exit disambiguation: rate-limit MUST win over the budget default."""

    def test_rate_limit_beats_budget(self):
        # 429 on a budgeted arm: an error to re-run, NOT a (kept-as-ok) budget wall.
        is_error, stop_reason = classify_codex_nonzero_exit(
            did_work=True, has_rollout_budget=True,
            stderr="stream disconnected: HTTP 429 Too Many Requests")
        assert is_error is True
        assert stop_reason == "rate_limited"

    def test_clean_budget_exit_is_budget_exhausted(self):
        is_error, stop_reason = classify_codex_nonzero_exit(
            did_work=True, has_rollout_budget=True, stderr="")
        assert is_error is False
        assert stop_reason == "budget_exhausted"

    def test_generic_error_when_no_budget(self):
        is_error, stop_reason = classify_codex_nonzero_exit(
            did_work=True, has_rollout_budget=False, stderr="segfault in tool")
        assert is_error is True
        assert stop_reason is None

    def test_no_work_does_not_earn_budget_credit(self):
        # exited before doing anything: a real error, not a budget truncation.
        is_error, stop_reason = classify_codex_nonzero_exit(
            did_work=False, has_rollout_budget=True, stderr="")
        assert is_error is True
        assert stop_reason is None
