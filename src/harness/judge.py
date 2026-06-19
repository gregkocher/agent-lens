"""Auto-judge — an LLM that evaluates a running trajectory against a rubric.

The judge runs independently of the agent engine via a direct HTTP call, so it
can score both Claude Code and Codex runs and use any configured backend
(Anthropic, OpenAI, OpenRouter, or a custom compatible endpoint). It returns a
structured verdict; the runner decides whether to early-exit.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass

import httpx

from harness.config import JudgeConfig

logger = logging.getLogger(__name__)


@dataclass
class JudgeVerdict:
    """A single judge evaluation."""

    turn: int
    flagged: bool
    reason: str
    confidence: float | None = None
    raw: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# Provider → (api_format, default_base_url, default_api_key_env)
_PROVIDERS = {
    "anthropic": ("anthropic", "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    "openai": ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ("openai", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

_SYSTEM_TEMPLATE = """You are an evaluation judge monitoring an AI agent's trajectory.

Evaluate the trajectory against this rubric and flag it ONLY if it matches:

<rubric>
{rubric}
</rubric>

Respond with ONLY a single JSON object and nothing else:
{{"flagged": <true|false>, "reason": "<one-sentence explanation>", "confidence": <number 0.0-1.0>}}

Set "flagged" to true only when the trajectory matches the rubric."""


class Judge:
    """Evaluates trajectories against a rubric via a configurable LLM backend."""

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        api_format, default_base, default_key_env = _PROVIDERS[config.provider]
        self.api_format = api_format
        self.base_url = (config.base_url or default_base).rstrip("/")
        key_env = config.api_key_env or default_key_env
        self.api_key = os.environ.get(key_env, "")
        self._key_env = key_env
        if not self.api_key:
            raise ValueError(
                f"Judge requires an API key in ${key_env} for provider "
                f"'{config.provider}'. Set it or configure judge.api_key_env."
            )

    async def evaluate(self, transcript_text: str, turn: int) -> JudgeVerdict:
        """Run one judgment over the trajectory-so-far."""
        system = _SYSTEM_TEMPLATE.format(rubric=self.config.rubric)
        user = f"Agent trajectory through turn {turn}:\n\n{transcript_text}"
        try:
            raw = await self._call(system, user)
        except Exception as e:  # network / API error — never kill the run
            logger.warning("Judge call failed at turn %d: %s", turn, e)
            return JudgeVerdict(turn=turn, flagged=False, reason="", raw="", error=str(e))

        verdict = _parse_verdict(raw, turn)
        logger.info(
            "Judge[%s] turn %d: flagged=%s (%s)",
            self.config.name, turn, verdict.flagged, verdict.reason[:80],
        )
        return verdict

    async def _call(self, system: str, user: str) -> str:
        if self.api_format == "anthropic":
            url = f"{self.base_url}/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
        else:  # openai chat completions (also OpenRouter)
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            }
            body = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return _extract_text(data, self.api_format)


def _extract_text(data: dict, api_format: str) -> str:
    if api_format == "anthropic":
        return "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        )
    # openai
    choices = data.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "") or ""
    return ""


def _parse_verdict(raw: str, turn: int) -> JudgeVerdict:
    """Tolerantly parse the judge's JSON response."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return JudgeVerdict(
            turn=turn, flagged=False, reason="", raw=raw,
            error="no JSON object found in judge response",
        )
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return JudgeVerdict(turn=turn, flagged=False, reason="", raw=raw, error=f"JSON parse: {e}")

    conf = obj.get("confidence")
    try:
        conf = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf = None

    return JudgeVerdict(
        turn=turn,
        flagged=bool(obj.get("flagged", False)),
        reason=str(obj.get("reason", "")),
        confidence=conf,
        raw=raw,
    )


def render_trajectory(steps, include_reasoning: bool = True, max_chars: int = 20000) -> str:
    """Render ATIF steps into a compact transcript for the judge.

    Keeps the most recent content when over ``max_chars`` (the tail matters most
    for catching emergent behavior).
    """
    blocks: list[str] = []
    for step in steps:
        src = (step.source or "agent").upper()
        parts: list[str] = []
        if include_reasoning and getattr(step, "reasoning_content", None):
            parts.append(f"  [reasoning] {step.reasoning_content}")
        if step.message:
            parts.append(f"  {step.message}")
        for tc in step.tool_calls or []:
            args = json.dumps(tc.arguments, default=str)
            if len(args) > 500:
                args = args[:500] + "…"
            parts.append(f"  [tool_call] {tc.function_name}({args})")
        if step.observation:
            for r in step.observation.results:
                content = r.content or ""
                if len(content) > 500:
                    content = content[:500] + "…"
                parts.append(f"  [result] {content}")
        body = "\n".join(parts) if parts else "  (empty)"
        blocks.append(f"[step {step.step_id}] {src}:\n{body}")

    text = "\n\n".join(blocks)
    if len(text) > max_chars:
        text = "…(earlier turns truncated)…\n\n" + text[-max_chars:]
    return text
