"""Tests for the auto-judge: verdict parsing, rendering, backend resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from harness.config import JudgeConfig
from harness.judge import (
    Judge,
    _extract_text,
    _parse_verdict,
    render_trajectory,
)


def _cfg(**kw):
    base = {"model": "m", "rubric": "flag bad things"}
    base.update(kw)
    return JudgeConfig(**base)


class TestParseVerdict:
    def test_clean_json(self):
        v = _parse_verdict('{"flagged": true, "reason": "did X", "confidence": 0.9}', 5)
        assert v.flagged is True
        assert v.reason == "did X"
        assert v.confidence == 0.9
        assert v.turn == 5

    def test_json_embedded_in_prose(self):
        v = _parse_verdict('Here is my verdict:\n{"flagged": false, "reason": "fine"}\nDone.', 1)
        assert v.flagged is False
        assert v.reason == "fine"

    def test_no_json(self):
        v = _parse_verdict("I cannot answer", 2)
        assert v.flagged is False
        assert v.error is not None

    def test_malformed_json(self):
        v = _parse_verdict('{"flagged": true, oops}', 3)
        assert v.flagged is False
        assert "JSON parse" in v.error

    def test_bad_confidence_ignored(self):
        v = _parse_verdict('{"flagged": true, "reason": "x", "confidence": "high"}', 1)
        assert v.flagged is True
        assert v.confidence is None


class TestExtractText:
    def test_anthropic(self):
        data = {"content": [{"type": "text", "text": "hello"}, {"type": "thinking", "text": "z"}]}
        assert _extract_text(data, "anthropic") == "hello"

    def test_openai(self):
        data = {"choices": [{"message": {"content": "hi there"}}]}
        assert _extract_text(data, "openai") == "hi there"

    def test_openai_empty(self):
        assert _extract_text({"choices": []}, "openai") == ""


class TestRenderTrajectory:
    def _steps(self):
        return [
            SimpleNamespace(
                step_id=1, source="agent", message="reading file",
                reasoning_content="let me think", tool_calls=[
                    SimpleNamespace(function_name="Read", arguments={"file_path": "/x"})
                ], observation=None,
            ),
            SimpleNamespace(
                step_id=2, source="agent", message="", reasoning_content=None,
                tool_calls=None,
                observation=SimpleNamespace(results=[SimpleNamespace(content="file contents")]),
            ),
        ]

    def test_render_includes_parts(self):
        out = render_trajectory(self._steps())
        assert "[step 1] AGENT" in out
        assert "[reasoning] let me think" in out
        assert "[tool_call] Read(" in out
        assert "[result] file contents" in out

    def test_reasoning_excluded(self):
        out = render_trajectory(self._steps(), include_reasoning=False)
        assert "let me think" not in out

    def test_truncation_keeps_tail(self):
        steps = [
            SimpleNamespace(step_id=i, source="agent", message="x" * 1000,
                            reasoning_content=None, tool_calls=None, observation=None)
            for i in range(50)
        ]
        out = render_trajectory(steps, max_chars=2000)
        assert out.startswith("…(earlier turns truncated)…")
        assert len(out) < 2200


class TestJudgeBackendResolution:
    def test_anthropic_default(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        j = Judge(_cfg(provider="anthropic"))
        assert j.api_format == "anthropic"
        assert j.base_url == "https://api.anthropic.com"

    def test_openrouter_uses_openai_format(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        j = Judge(_cfg(provider="openrouter"))
        assert j.api_format == "openai"
        assert "openrouter.ai" in j.base_url

    def test_custom_base_url_and_key_env(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret")
        j = Judge(_cfg(provider="openai", base_url="https://x.test/v1", api_key_env="MY_KEY"))
        assert j.base_url == "https://x.test/v1"
        assert j.api_key == "secret"

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="requires an API key"):
            Judge(_cfg(provider="anthropic"))
