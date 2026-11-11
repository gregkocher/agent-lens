"""Codex retry notices must not mark a run as failed."""
from harness.engines.codex import is_transient_codex_error


def test_reconnect_notices_are_transient():
    assert is_transient_codex_error(
        "Reconnecting... 1/5 (stream disconnected before completion: Internal server error)")
    assert is_transient_codex_error("Reconnecting... 5/5 (unexpected status 502 Bad Gateway)")


def test_other_errors_are_not_transient():
    assert not is_transient_codex_error("stream disconnected before completion: Internal server error")
    assert not is_transient_codex_error("unexpected status 401 Unauthorized")
    assert not is_transient_codex_error("")
