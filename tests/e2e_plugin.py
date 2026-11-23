"""End-to-end test of astrbot_plugin_deepseek_research on the real codex_astrbot
binding, with a scripted model, a mock DeepSeek API and a local stand-in sandbox.

Usage, from an AstrBot source tree with codex_astrbot and aiohttp installed:
    PYTHONPATH=. python <plugin>/tests/e2e_plugin.py <plugin parent dir> <work dir>
The plugin directory must be named astrbot_plugin_deepseek_research.
"""

import asyncio
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from aiohttp import web

PLUGIN_PARENT, WORK = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(PLUGIN_PARENT))
WORK.mkdir(parents=True, exist_ok=True)

import astrbot.core.computer.computer_client as cc
import astrbot_plugin_deepseek_research.main as plugin_main
from astrbot.core.agent.runners.codex.native import CodexEngine

SANDBOX_ROOT = str(WORK / "sandbox_tmp" / "ds_research")
plugin_main.SANDBOX_ROOT = SANDBOX_ROOT

LOG = {"ds": [], "llm_tools": [], "llm_inputs": [], "tool_outputs": []}
SCENARIO = {"name": "happy", "delay": 0.0}

HAPPY = [
    ("deepseek_fetch", {"url": "https://example.com/a"}),
    ("deepseek_search", {"query": "测试 关键词", "sites": ["tieba.baidu.com"]}),
    ("tmp_list", {}),
    ("tmp_read", {"path": "search/01.md", "limit": 200}),
    ("deepseek_fetch", {"url": "https://www.youtube.com/watch?v=x"}),
    ("deepseek_fetch", {"url": "https://example.com/a", "offset": 10}),
    ("tmp_read", {"path": "../../escape.txt"}),
    (
        "apply_patch",
        {
            "patch": "*** Begin Patch\n*** Add File: notes.md\n+# Notes\n+- todo\n*** End Patch"
        },
    ),
    (
        "apply_patch",
        {
            "patch": "*** Begin Patch\n*** Update File: notes.md\n@@ # Notes\n-- todo\n+- found [1]\n"
            "*** Add File: draft.md\n+draft\n*** End Patch"
        },
    ),
    # A hunk that does not match: nothing of this patch may be written.
    (
        "apply_patch",
        {
            "patch": "*** Begin Patch\n*** Add File: should_not_exist.md\n+x\n"
            "*** Update File: notes.md\n-no such line\n+y\n*** End Patch"
        },
    ),
    (
        "apply_patch",
        {
            "patch": "*** Begin Patch\n*** Update File: draft.md\n*** Move to: final/draft.md\n-draft\n+final\n*** End Patch"
        },
    ),
    (
        "apply_patch",
        {
            "patch": "*** Begin Patch\n*** Add File: ok_before_escape.md\n+x\n*** Add File: ../../escape.txt\n+x\n*** End Patch"
        },
    ),
    ("tmp_read", {"path": "notes.md"}),
]
FINAL = (
    "结论：测试通过 [1]。\n\n来源：\n[1] Page A - https://example.com/a (2025-01-01)"
)


def brain(n_outputs: int, last_output: str):
    if last_output:
        LOG["tool_outputs"].append(last_output)
    if SCENARIO["name"] == "happy":
        if n_outputs < len(HAPPY):
            return HAPPY[n_outputs]
        return FINAL
    return ("deepseek_search", {"query": f"loop {n_outputs}"})


# ------------------------------------------------------------- mock LLM wires


def _sse(ev: dict, named: bool) -> bytes:
    head = f"event: {ev['type']}\n" if named else ""
    return f"{head}data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode()


async def responses(req):
    body = await req.json()
    LOG["llm_tools"].append(
        sorted(
            f"{t['name']}.{x['name']}"
            if t.get("type") == "namespace"
            else t.get("name") or t.get("type")
            for t in body.get("tools") or []
            for x in (t.get("tools") or [{"name": ""}])
        )
    )
    items = body.get("input") or []
    LOG["llm_inputs"].append(json.dumps(items, ensure_ascii=False))
    outs = [i for i in items if i.get("type") == "function_call_output"]
    last = str(outs[-1].get("output")) if outs else ""
    await asyncio.sleep(SCENARIO["delay"])
    step = brain(len(outs), last)
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(req)
    await resp.write(_sse({"type": "response.created", "response": {"id": "r"}}, True))
    if isinstance(step, tuple):
        item = {
            "type": "function_call",
            "id": f"fc{time.time_ns()}",
            "call_id": f"call{time.time_ns()}",
            "name": step[0],
            "namespace": "research",
            "arguments": json.dumps(step[1], ensure_ascii=False),
            "status": "completed",
        }
    else:
        item = {
            "type": "message",
            "role": "assistant",
            "id": "m",
            "content": [{"type": "output_text", "text": step}],
        }
    await resp.write(_sse({"type": "response.output_item.done", "item": item}, True))
    await resp.write(
        _sse(
            {
                "type": "response.completed",
                "response": {
                    "id": "r",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            },
            True,
        )
    )
    await resp.write_eof()
    return resp


async def chat(req):
    body = await req.json()
    LOG["llm_tools"].append(
        sorted(t["function"]["name"] for t in body.get("tools") or [])
    )
    msgs = body.get("messages") or []
    LOG["llm_inputs"].append(json.dumps(msgs, ensure_ascii=False))
    outs = [m for m in msgs if m.get("role") == "tool"]
    last = str(outs[-1].get("content")) if outs else ""
    await asyncio.sleep(SCENARIO["delay"])
    step = brain(len(outs), last)
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(req)
    if isinstance(step, tuple):
        delta = {
            "role": "assistant",
            "tool_calls": [
                {
                    "index": 0,
                    "id": f"call{time.time_ns()}",
                    "type": "function",
                    "function": {
                        "name": f"research__{step[0]}",
                        "arguments": json.dumps(step[1], ensure_ascii=False),
                    },
                }
            ],
        }
        finish = "tool_calls"
    else:
        delta, finish = {"role": "assistant", "content": step}, "stop"
    await resp.write(
        _sse(
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            False,
        )
    )
    await resp.write(
        _sse(
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            False,
        )
    )
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


# ------------------------------------------------------------ mock DeepSeek


async def ds_search(req):
    body = await req.json()
    LOG["ds"].append(("search", body))
    if req.headers.get("Authorization") != "Bearer test-key":
        return web.json_response({"error": {"message": "bad key"}}, status=401)
    results = [
        {
            "rerank_score": 1.2,
            "title": f"帖子 {i}",
            "url": f"https://tieba.baidu.com/p/{i}",
            "snippet": "s",
            "summary": f"summary {i}",
            "content": f"正文 {i} " * 50 if body.get("need_content") else "",
            "site_name": "百度贴吧",
            "site_icon": None,
            "date": "2025-01-01T00:00:00+00:00",
            "authority": 2,
        }
        for i in range(1, 4)
    ]
    return web.json_response({"results": results})


async def ds_fetch(req):
    body = await req.json()
    LOG["ds"].append(("fetch", body))
    if "youtube.com" in body["url"]:
        return web.json_response(
            {"error": {"message": "failed to fetch URL", "type": "not_found_error"}},
            status=404,
        )
    return web.json_response(
        {"title": "Page A", "content": "0123456789" + "页面内容" * 3000}
    )


# ------------------------------------------------------- stand-in sandbox


class LocalPython:
    async def exec(self, code, timeout=30, **_):
        # The plugin's sandbox code, with /tmp mapped into the work dir.
        p = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {"output": p.stdout, "error": p.stderr, "success": p.returncode == 0}


class FakeBooter:
    python = LocalPython()


async def fake_get_booter(context, session_id):
    return FakeBooter()


cc.get_booter = fake_get_booter


class FakeContext:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_config(self, umo=None):
        return self.cfg


class FakeEvent:
    unified_msg_origin = "test:FriendMessage:42"


def chat_cfg(port, wire, tool_call_timeout):
    provider = {
        "id": "deepseek",
        "name": "DeepSeek",
        "base_url": f"http://127.0.0.1:{port}/v1",
        "api_key": "x",
        "wire_api": wire,
        "models": [{"slug": "mock-ds"}],
    }
    return {
        "provider_settings": {"computer_use_runtime": "sandbox"},
        "agent_runner": {
            "runner_type": "codex",
            "config": {
                "codex_home": str(WORK / "codex_home"),
                "model_providers": [provider],
                "model_provider": "deepseek",
                "model": "mock-ds",
                "web_search": True,
                "tool_call_timeout": tool_call_timeout,
            },
        },
    }


async def run_tool(plugin, **kw):
    return await plugin.deepseek_research(FakeEvent(), **kw)


async def main():
    app = web.Application()
    app.router.add_post("/v1/responses", responses)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/v1/tools/search", ds_search)
    app.router.add_post("/v1/tools/fetch", ds_fetch)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    ok = True

    def check(cond, label):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + label)

    plugin_cfg = {
        "deepseek_api_key": "test-key",
        "deepseek_base_url": f"http://127.0.0.1:{port}",
        "max_tool_calls": 30,
    }
    for wire in ("responses", "chat"):
        shutil.rmtree(WORK / "sandbox_tmp", ignore_errors=True)
        for v in LOG.values():
            v.clear()
        SCENARIO.update(name="happy", delay=0.0)
        plugin = plugin_main.DeepSeekResearchPlugin(
            FakeContext(chat_cfg(port, wire, 120)), plugin_cfg
        )
        t0 = time.monotonic()
        out = await run_tool(
            plugin,
            keywords="测试",
            instruction="总结",
            time_range="30d",
            priority_urls=["https://example.com/a"],
        )
        print(f"--- [{wire}] result ({time.monotonic() - t0:.1f}s):\n{out}\n---")
        want = [
            "research.apply_patch",
            "research.deepseek_fetch",
            "research.deepseek_search",
            "research.tmp_list",
            "research.tmp_read",
        ]
        if wire == "chat":
            want = [w.replace(".", "__") for w in want]
        check(
            all(t == want for t in LOG["llm_tools"]) and LOG["llm_tools"],
            f"[{wire}] model sees only the 5 tools",
        )
        searches = [b for k, b in LOG["ds"] if k == "search"]
        check(
            searches
            and "site:tieba.baidu.com" in searches[0]["query"]
            and "after:" in searches[0]["query"]
            and searches[0]["need_content"] is True,
            f"[{wire}] search query got sites + time range: {searches and searches[0]['query']}",
        )
        fetches = [b["url"] for k, b in LOG["ds"] if k == "fetch"]
        check(
            fetches.count("https://example.com/a") == 1,
            f"[{wire}] page fetched once, re-read from cache ({fetches})",
        )
        check(
            any("could not fetch" in o for o in LOG["tool_outputs"]),
            f"[{wire}] blocked fetch explained",
        )
        check(
            any("outside the task directory" in o for o in LOG["tool_outputs"]),
            f"[{wire}] path escape refused",
        )
        check(
            any(
                "search/01.md" in o and "pages/01-example.com.md" in o
                for o in LOG["tool_outputs"]
            ),
            f"[{wire}] tmp_list shows saved files",
        )
        dirs = list(Path(SANDBOX_ROOT).iterdir())
        files = (
            sorted(
                str(p.relative_to(dirs[0])) for p in dirs[0].rglob("*") if p.is_file()
            )
            if dirs
            else []
        )
        check(
            files
            == [
                "final/draft.md",
                "notes.md",
                "pages/01-example.com.md",
                "report.md",
                "search/01.md",
            ],
            f"[{wire}] sandbox files {files}",
        )
        if dirs:
            notes = (dirs[0] / "notes.md").read_text()
            moved = (dirs[0] / "final/draft.md").read_text()
            check(
                notes == "# Notes\n- found [1]\n" and moved == "final\n",
                f"[{wire}] patch results {notes!r} {moved!r}",
            )
        check(
            any(
                "could not locate hunk" in o and "nothing written" in o
                for o in LOG["tool_outputs"]
            ),
            f"[{wire}] failed hunk reported, patch not applied",
        )
        check(
            sum("outside the task directory" in o for o in LOG["tool_outputs"]) == 2,
            f"[{wire}] read and patch outside the directory refused",
        )
        check(
            not (WORK / "sandbox_tmp" / "escape.txt").exists(),
            f"[{wire}] nothing written outside",
        )
        check(
            out.startswith("结论：测试通过") and "/ds_research/" in out,
            f"[{wire}] report + material path returned",
        )
        engine = next(iter(CodexEngine._instances.values()))
        check(not engine.pumps, f"[{wire}] no thread pump left ({list(engine.pumps)})")

    # Model that never stops: refused past max_tool_calls, cut at the hard deadline.
    for v in LOG.values():
        v.clear()
    SCENARIO.update(name="stubborn", delay=0.2)
    plugin = plugin_main.DeepSeekResearchPlugin(
        FakeContext(chat_cfg(port, "chat", 45)), {**plugin_cfg, "max_tool_calls": 4}
    )
    t0 = time.monotonic()
    out = await run_tool(plugin, keywords="loop")
    took = time.monotonic() - t0
    print(f"--- [timeout] result ({took:.1f}s):\n{out}\n---")
    check(28 <= took <= 36, f"[timeout] ends at the 30 s budget ({took:.1f}s)")
    check(
        out.startswith("调研未完成（超出时间预算）"),
        "[timeout] partial result says why",
    )
    check(
        len([1 for k, _ in LOG["ds"] if k == "search"]) == 4,
        "[timeout] max_tool_calls enforced",
    )
    check(
        any("Time is almost up" in i for i in LOG["llm_inputs"]),
        "[timeout] wrap-up note steered in",
    )
    engine = next(iter(CodexEngine._instances.values()))
    check(not engine.pumps, "[timeout] thread released")

    # AstrBot with tool tasks: tool_call_timeout no longer caps the budget.
    for v in LOG.values():
        v.clear()
    sys.modules["astrbot.core.tools.tool_tasks"] = type(sys)(
        "astrbot.core.tools.tool_tasks"
    )
    plugin = plugin_main.DeepSeekResearchPlugin(
        FakeContext(chat_cfg(port, "chat", 20)),
        {**plugin_cfg, "max_tool_calls": 4, "time_budget_s": 38},
    )
    t0 = time.monotonic()
    out = await run_tool(plugin, keywords="loop")
    took = time.monotonic() - t0
    del sys.modules["astrbot.core.tools.tool_tasks"]
    check(
        36 <= took <= 44,
        f"[tool tasks] budget is time_budget_s, not tool_call_timeout ({took:.1f}s)",
    )

    # Cancelled from outside (the model cancels the tool task, or a user stop).
    SCENARIO.update(name="stubborn", delay=1.0)
    task = asyncio.create_task(run_tool(plugin, keywords="cancel me"))
    await asyncio.sleep(4)
    loaded = list(engine.pumps)
    task.cancel()
    with __import__("contextlib").suppress(asyncio.CancelledError):
        await task
    await asyncio.sleep(1)
    still = [t for t in loaded if await engine.rt.is_loaded(t)]
    check(
        loaded and not still and not engine.pumps, f"[cancel] thread {loaded} shut down"
    )

    await CodexEngine.shutdown_all()
    await plugin.terminate()
    await runner.cleanup()
    print("ALL PASS" if ok else "SOME FAILED")


asyncio.run(main())
