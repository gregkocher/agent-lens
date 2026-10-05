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
from harness.judge_budget import shrink_largest

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
    render_info: dict | None = None  # what (if anything) was cut from the judge input

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


def _get(obj, key, default=None):
    """Field access for ATIF steps given as model objects or as JSON dicts."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _reasoning_line(step) -> str | None:
    text = _get(step, "reasoning_content")
    kind = (_get(step, "extra") or {}).get("reasoning_kind") if isinstance(_get(step, "extra"), dict) else None
    if text:
        return f"  [reasoning{' (summary)' if kind == 'summary' else ''}] {text}"
    if kind == "encrypted":
        return "  [reasoning] (present but encrypted by the provider; not readable)"
    return None


def render_trajectory_with_info(steps, include_reasoning: bool = True,
                                max_chars: int = 750_000) -> tuple[str, dict]:
    """Render the FULL trajectory for the judge: every reasoning block, message, tool
    call and tool output. Over ``max_chars``, the largest tool outputs are shortened
    first (judge_budget); only if that is not enough are the earliest steps dropped
    (keeping the most recent ones). Returns (text, info about what was cut)."""
    heads: list[tuple[str, list[str]]] = []
    results: list[list[str]] = []
    for step in steps:
        src = (_get(step, "source") or "agent").upper()
        parts: list[str] = []
        if include_reasoning:
            line = _reasoning_line(step)
            if line:
                parts.append(line)
        if _get(step, "message"):
            parts.append(f"  {_get(step, 'message')}")
        for tc in _get(step, "tool_calls") or []:
            args = json.dumps(_get(tc, "arguments"), default=str)
            parts.append(f"  [tool_call] {_get(tc, 'function_name')}({args})")
        heads.append((f"[step {_get(step, 'step_id')}] {src}:", parts))
        obs = _get(step, "observation")
        results.append([(_get(r, "content") or "") for r in (_get(obs, "results") or [])] if obs else [])

    def assemble(res: list[list[str]]) -> list[str]:
        blocks = []
        for (label, parts), rs in zip(heads, res):
            lines = parts + [f"  [result] {c}" for c in rs]
            blocks.append(label + "\n" + ("\n".join(lines) if lines else "  (empty)"))
        return blocks

    info = {"budget": max_chars, "truncated": False, "tool_outputs_shortened": 0,
            "tool_output_chars_omitted": 0, "steps_omitted": 0}
    text = "\n\n".join(assemble(results))
    if len(text) > max_chars:
        flat = [c for rs in results for c in rs]
        shrunk = shrink_largest(flat, len(text) - max_chars)
        it = iter(shrunk.texts)
        results = [[next(it) for _ in rs] for rs in results]
        info.update(truncated=True, tool_outputs_shortened=shrunk.shortened,
                    tool_output_chars_omitted=shrunk.chars_omitted)
        blocks = assemble(results)
        text = "\n\n".join(blocks)
        if len(text) > max_chars:  # last resort: keep the most recent steps
            kept: list[str] = []
            used = 0
            for b in reversed(blocks):
                if used + len(b) + 2 > max_chars - 60:
                    break
                kept.insert(0, b)
                used += len(b) + 2
            info["steps_omitted"] = len(blocks) - len(kept)
            text = "…(earlier turns truncated)…\n\n" + "\n\n".join(kept)
            if not kept:  # a single step larger than the whole budget
                text = "…(earlier turns truncated)…\n\n" + text[-max_chars:]
    info["rendered_chars"] = len(text)
    return text, info


def render_trajectory(steps, include_reasoning: bool = True, max_chars: int = 750_000) -> str:
    """Full-trajectory judge input (see ``render_trajectory_with_info``)."""
    return render_trajectory_with_info(steps, include_reasoning, max_chars)[0]
