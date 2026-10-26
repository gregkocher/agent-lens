"""Faithful Codex resumes: undo what ``codex exec resume`` adds to the conversation.

``codex exec resume <id> PROMPT`` replays the stored rollout and then appends a fresh
developer (permissions) message, a fresh environment-context message and PROMPT. To make
the model see exactly the stored history, resumes pass ``RESUME_MARKER`` as PROMPT and a
capture-proxy intercept removes the marker message plus the messages Codex re-injected
right before it, on every request of the resumed session.

The marker is plain text that reveals nothing about the harness: Codex also writes it to
the rollout in the agent's own ``~/.codex/sessions``, which the agent can read, and if a
strip ever fails the model sees it. It is matched only as the exact text of a user message
after the resumed history, so the same words inside the history are never touched.
"""

from __future__ import annotations

import json

RESUME_MARKER = "Continue."


def is_resume_marker(item: dict) -> bool:
    """True if ``item`` is the resume prompt Codex appended (a user message whose text is the marker)."""
    if item.get("type") != "message" or item.get("role") != "user":
        return False
    content = item.get("content")
    if isinstance(content, str):
        return content.strip() == RESUME_MARKER
    return any(isinstance(p, dict) and str(p.get("text", "")).strip() == RESUME_MARKER for p in content or [])


def strip_resume_additions(items: list[dict], keep: int) -> list[dict]:
    """Drop the marker message and the context messages Codex re-injected before it.
    ``keep`` = number of items of the resumed history (never touched)."""
    m = next((i for i in range(keep, len(items)) if is_resume_marker(items[i])), None)
    if m is None:
        return items
    start = m
    while start > keep and items[start - 1].get("type") == "message" and (
            items[start - 1].get("role") == "developer"
            or "<environment_context>" in json.dumps(items[start - 1].get("content"))):
        start -= 1
    return items[:start] + items[m + 1:]


class ResumeStrip:
    """Capture-proxy intercept that only strips resume additions (replays)."""

    def __init__(self, keep: int):
        self.keep = keep

    async def __call__(self, request_data: dict, request_index: int) -> None:
        if isinstance(request_data.get("input"), list):
            request_data["input"] = strip_resume_additions(request_data["input"], self.keep)
        return None
