"""Tests for Codex/OpenAI capture + resample format handling (no network)."""

from __future__ import annotations

from harness.proxy import _detect_api_format, _parse_openai_responses_sse
from harness.resample import (
    _build_headers,
    _detect_format,
    _prepare_request,
    _summarize_response,
)


class TestProxyFormatDetection:
    def test_anthropic_path(self):
        assert _detect_api_format("/v1/messages") == "anthropic"

    def test_openai_path(self):
        assert _detect_api_format("/v1/responses") == "openai_responses"

    def test_other_path(self):
        assert _detect_api_format("/v1/models") is None


class TestOpenAISSEParsing:
    def test_extracts_usage_and_model(self):
        sse = (
            'event: response.completed\n'
            'data: {"type":"response.completed","response":{"model":"gpt-5.4",'
            '"usage":{"input_tokens":100,"output_tokens":20,'
            '"input_tokens_details":{"cached_tokens":80},'
            '"output_tokens_details":{"reasoning_tokens":5}}}}\n\n'
        ).encode()
        meta = _parse_openai_responses_sse(sse)
        assert meta["response_model"] == "gpt-5.4"
        assert meta["openai_usage"]["input_tokens"] == 100
        assert meta["openai_usage"]["input_tokens_details"]["cached_tokens"] == 80

    def test_ignores_done_sentinel(self):
        assert _parse_openai_responses_sse(b"data: [DONE]\n\n") == {}


class TestResampleFormatHandling:
    def test_detect_openai_by_path(self):
        assert _detect_format("https://api.openai.com/v1/responses", {}) == "openai_responses"

    def test_detect_openai_by_body(self):
        assert _detect_format("", {"input": [], "model": "x"}) == "openai_responses"

    def test_detect_anthropic(self):
        assert _detect_format("https://api.anthropic.com/v1/messages", {"messages": []}) == "anthropic"

    def test_prepare_openai_keeps_context_fields(self):
        # Anthropic-only cleanup must not touch openai requests.
        req = {"input": [], "metadata": {"x": 1}, "model": "gpt-5.4"}
        out = _prepare_request(req, fmt="openai_responses")
        assert out["stream"] is False
        assert "metadata" in out  # not stripped for openai

    def test_prepare_anthropic_strips_sdk_fields(self):
        req = {"messages": [], "context_management": {}, "metadata": {}}
        out = _prepare_request(req, fmt="anthropic")
        assert "context_management" not in out
        assert "metadata" not in out

    def test_openai_uses_bearer_auth(self):
        h = _build_headers({"x-api-key": "old"}, "KEY", "https://api.openai.com/v1/responses", "openai_responses")
        assert h["Authorization"] == "Bearer KEY"
        assert "x-api-key" not in h

    def test_anthropic_uses_x_api_key(self):
        h = _build_headers({"Authorization": "old"}, "KEY", "https://api.anthropic.com/v1/messages", "anthropic")
        assert h["x-api-key"] == "KEY"
        assert "Authorization" not in h

    def test_summarize_openai(self):
        blocks, toks = _summarize_response(
            {"output": [{"type": "reasoning"}, {"type": "message"}], "usage": {"output_tokens": 7}},
            "openai_responses",
        )
        assert blocks == ["reasoning", "message"]
        assert toks == 7
