"""Faithful Codex resumes: undo what ``codex exec resume`` adds to the conversation.

``codex exec resume <id> PROMPT`` replays the stored rollout and then appends a fresh
developer (permissions) message, a fresh environment-context message and PROMPT. To make
the model see exactly the stored history, resumes pass ``RESUME_MARKER`` as PROMPT and a
capture-proxy intercept removes the marker message plus the messages Codex re-injected
right before it, on every request of the resumed session.
"""

from __future__ import annotations

import json

RESUME_MARKER = "__agentlens_branch_resume__"


def strip_resume_additions(items: list[dict], keep: int) -> list[dict]:
    """Drop the marker message and the context messages Codex re-injected before it.
    ``keep`` = number of items of the resumed history (never touched)."""
    m = next((i for i, it in enumerate(items)
              if it.get("type") == "message" and RESUME_MARKER in json.dumps(it)), None)
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
