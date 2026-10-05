"""Context window / max output of OpenRouter models, for Codex.

Codex ships no profile for OpenRouter models: it warns "model metadata not found" and
assumes a 272k window (258,400 usable), so it would compact at the wrong point or let
requests overflow a smaller provider window (e.g. gpt-oss-120b on Cerebras). The real
limits come from OpenRouter's endpoint list: the pinned provider's endpoint, or the
smallest across all endpoints when unpinned (any of them may serve a request).
Codex 0.142 caps the window at 272k, so larger provider windows change nothing.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

_CACHE: dict[tuple[str, tuple[str, ...]], dict | None] = {}


def _pick(endpoints: list[dict], provider_order: list[str] | None) -> dict | None:
    if provider_order:
        want = provider_order[0].lower()
        for e in endpoints:
            tag = (e.get("tag") or "").lower()
            if tag == want or tag.split("/")[0] == want:
                return {"context_window": e.get("context_length"),
                        "max_output_tokens": e.get("max_completion_tokens"),
                        "endpoint": e.get("tag")}
        return None
    ctx = [e["context_length"] for e in endpoints if e.get("context_length")]
    out = [e["max_completion_tokens"] for e in endpoints if e.get("max_completion_tokens")]
    if not ctx:
        return None
    return {"context_window": min(ctx), "max_output_tokens": min(out) if out else None,
            "endpoint": "min over all endpoints"}


def openrouter_limits(model: str, provider_order: list[str] | None = None,
                      base_url: str = "https://openrouter.ai/api/v1",
                      timeout: float = 20.0) -> dict | None:
    """``{"context_window", "max_output_tokens", "endpoint"}`` or None if unavailable."""
    key = (model, tuple(provider_order or ()))
    if key in _CACHE:
        return _CACHE[key]
    headers = {}
    if os.environ.get("OPENROUTER_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OPENROUTER_API_KEY']}"
    result = None
    try:
        r = httpx.get(f"{base_url}/models/{model}/endpoints", headers=headers, timeout=timeout)
        r.raise_for_status()
        endpoints = (r.json().get("data") or {}).get("endpoints") or []
        result = _pick(endpoints, provider_order)
        if result is None:
            logger.warning("No OpenRouter endpoint matches %s for %s; Codex keeps its default "
                           "context window", provider_order, model)
    except Exception as exc:  # network or schema trouble: fall back to Codex defaults
        logger.warning("Could not fetch OpenRouter limits for %s (%s); Codex keeps its default "
                       "context window", model, exc)
    _CACHE[key] = result
    return result


def _override(overrides: list[str], key: str) -> str | None:
    for o in overrides or []:
        k, _, v = str(o).partition("=")
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return None


def resolve_sampling(rc) -> dict:
    """The sampling a run actually uses (recorded in run_meta.json["sampling"]).

    Codex engine only; claude_code returns an empty dict. ``temperature``/``top_p`` are
    only applied when the capture proxy runs (it injects them).
    """
    if rc.engine != "codex":
        return {}
    from harness.prefill import recommended_sampling

    effort = _override(rc.codex_config_overrides, "model_reasoning_effort") or rc.reasoning_effort
    t, p = rc.temperature, rc.top_p
    source = "config"
    if rc.use_recommended_sampling:
        rt, rp = recommended_sampling(rc.model)
        if (t is None and rt is not None) or (p is None and rp is not None):
            source = "config + model recommendation"
        t = rt if t is None else t
        p = rp if p is None else p
    if not rc.capture_api_requests:
        t = p = None
    return {"reasoning_effort": effort, "temperature": t, "top_p": p, "source": source}


def resolve_codex_limits(rc) -> dict | None:
    """Codex ``model_context_window``/``model_max_output_tokens`` for an OpenRouter run."""
    if rc.engine != "codex" or rc.provider != "openrouter" or not rc.codex_model_limits:
        return None
    lim = openrouter_limits(rc.model, rc.provider_order)
    if not lim:
        return None
    out = dict(lim)
    for key, field in (("model_context_window", "context_window"),
                       ("model_max_output_tokens", "max_output_tokens")):
        if _override(rc.codex_config_overrides, key) is not None:
            out[field] = None  # an explicit override wins
    return out
