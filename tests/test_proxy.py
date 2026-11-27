"""DeepSeek API calls follow AstrBot's proxy setting (http_proxy / no_proxy)."""

import sys
from pathlib import Path

import pytest
from aiohttp import web

# The package's parent, so research.py imports as part of the plugin.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from astrbot_plugin_deepseek_research.research import DeepSeekError, DeepSeekTools


@pytest.mark.asyncio
async def test_astrbots_proxy_setting_is_followed(monkeypatch):
    calls = []

    async def search(req):
        calls.append(await req.json())
        return web.json_response({"results": []})

    app = web.Application()
    app.router.add_post("/v1/tools/search", search)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = DeepSeekTools("k", f"http://127.0.0.1:{port}")
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    try:
        # A proxy nobody listens on: going through it fails.
        for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            monkeypatch.setenv(name, "http://127.0.0.1:9")
        monkeypatch.setenv("no_proxy", "")
        monkeypatch.setenv("NO_PROXY", "")
        with pytest.raises(DeepSeekError, match="network error"):
            await client.call("search", {"query": "x"})
        assert calls == []
        await client.close()
        # no_proxy for the host: straight there.
        monkeypatch.setenv("no_proxy", "127.0.0.1")
        monkeypatch.setenv("NO_PROXY", "127.0.0.1")
        assert await client.call("search", {"query": "x"}) == {"results": []}
        assert len(calls) == 1
    finally:
        await client.close()
        await runner.cleanup()


async def _no_sleep(_seconds):
    return None
