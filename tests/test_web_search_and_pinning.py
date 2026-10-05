"""web_search is actually disabled for Codex; the proxy pins providers and flags server tools."""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp import web

from harness.config import RunConfig, SessionConfig
from harness.engines.base import EngineRunSpec
from harness.engines.codex import CodexEngine
from harness.proxy import CaptureProxy, _server_tool_types


def _spec(**extra):
    return EngineRunSpec(prompt="p", model="thinkingmachines/inkling", cwd="/tmp", provider="openrouter", extra=extra)


def test_codex_disables_web_search_with_the_working_flag():
    argv = CodexEngine()._build_argv(_spec(), "p")
    assert 'web_search="disabled"' in argv
    assert "tools.web_search=false" not in argv          # ignored by codex 0.142.0
    assert 'web_search="disabled"' not in CodexEngine()._build_argv(_spec(codex_web_search=True), "p")


def test_server_tool_detection():
    tools = [{"type": "function", "name": "exec"}, {"type": "namespace", "name": "ns"},
             {"type": "web_search", "external_web_access": True}]
    assert _server_tool_types(tools, "openai_responses") == {"web_search"}
    assert _server_tool_types([{"name": "Read", "input_schema": {}}], "anthropic") == set()
    assert _server_tool_types([{"type": "web_search_20250305", "name": "web_search"}], "anthropic") == {"web_search_20250305"}


def _cfg(**kw):
    base = dict(model="thinkingmachines/inkling", work_dir="/tmp", engine="codex", provider="openrouter",
                sessions=[SessionConfig(session_index=1, prompt="x")])
    base.update(kw)
    return RunConfig(**base)


def test_provider_pin_validation():
    assert _cfg(provider_order=["together"]).provider_order == ["together"]
    with pytest.raises(ValueError, match="openrouter"):
        _cfg(provider="openai", model="gpt-5.5", provider_order=["together"])
    with pytest.raises(ValueError, match="capture_api_requests"):
        _cfg(provider_order=["together"], capture_api_requests=False)
    with pytest.raises(ValueError, match="at least one"):
        _cfg(provider_order=[])


def test_proxy_injects_provider_pin_and_records_server_tools(tmp_path):
    seen = []

    async def upstream(request):
        seen.append(await request.json())
        return web.Response(text='data: {"type":"response.completed","response":{"usage":{}}}\n\n',
                            content_type="text/event-stream")

    async def go():
        app = web.Application()
        app.router.add_post("/v1/responses", upstream)
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
        port = site._server.sockets[0].getsockname()[1]
        proxy = CaptureProxy(raw_dump_count=10, inject={"provider": {"order": ["together"], "allow_fallbacks": False}})
        pport = await proxy.start(f"http://127.0.0.1:{port}", tmp_path / "api_captures.jsonl")
        import httpx
        async with httpx.AsyncClient() as c:
            await c.post(f"http://127.0.0.1:{pport}/v1/responses",
                         json={"model": "m", "input": [], "tools": [{"type": "web_search"}, {"type": "function", "name": "f"}]})
        await proxy.stop(); await runner.cleanup()
        return proxy

    proxy = asyncio.run(go())
    assert seen[0]["provider"] == {"order": ["together"], "allow_fallbacks": False}
    assert proxy.server_tools_seen == {"web_search"}
    assert json.loads((tmp_path / "raw_dumps" / "request_001.json").read_text())["provider"]["order"] == ["together"]


def test_sweep_provider_order_reaches_run_config():
    from pathlib import Path
    from pipeline.run_trajectories import _build_run_config
    from tests.test_isolation_realism import _run_config, _sweep
    rc = _build_run_config(_sweep(agent_provider_order=["together"]), _run_config("codex"), 0.05, 1, Path("/tmp/ws/x"))
    assert rc.provider_order == ["together"]
