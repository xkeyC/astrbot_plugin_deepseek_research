"""DeepSeek web research run by a restricted Codex subagent.

One research task is one ephemeral Codex thread that sees only the tools
defined here: DeepSeek's Search and Fetch APIs and file access confined to
``/tmp/ds_research/<task_id>/`` in the chat's sandbox. The thread has no
execution environment, no web search of its own, no images and no memories.

The subagent calls the tools from code (code mode): search results and pages
come back to its program as JSON and are saved to the task directory, so the
program filters them and files the material, and only what it prints reaches
the model. Reading whole pages into the model's context made research slow
and expensive, above all on a local model.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
import time
import uuid
from datetime import date, timedelta
from typing import Any
from urllib.parse import urlparse

import aiohttp
from astrbot.api import logger
from astrbot.core.tools.computer_tools.apply_patch import (
    PatchError,
    apply_hunks,
    parse_patch,
)

JsonObject = dict[str, Any]

TERMINAL_EVENTS = ("task_complete", "turn_complete", "turn_aborted")
NAMESPACE = "research"
SANDBOX_ROOT = "/tmp/ds_research"
# DeepSeek rejects longer queries (operators included).
MAX_QUERY_CHARS = 128
# Pages are kept whole up to this size, in memory and in the sandbox.
MAX_PAGE_CHARS = 300_000
MAX_READ_CHARS = 12_000
# A file the subagent writes from code.
MAX_WRITE_CHARS = 1_000_000
# A file apply_patch may edit: it is read and rewritten whole.
MAX_PATCH_FILE_CHARS = 1_000_000
# Statuses DeepSeek asks clients to retry after a pause.
RETRY_STATUSES = (408, 429, 500, 502, 503, 504)
# The rest of the budget after this share is for writing the report.
WRAP_UP_AT = 0.75

# Restricts the thread to its dynamic tools: per-thread overrides of the
# engine's options (probed on codex 348075fad: view_image survives
# no_environment unless its feature is off; web_search follows the runner's
# setting otherwise).
THREAD_CONFIG: JsonObject = {
    # The tools are called from the subagent's program (run_code for an
    # open-weight model, exec otherwise), not one call per model step.
    "model_tool_mode": "code_mode_only",
    "web_search": "disabled",
    "features.view_image": False,
    "features.image_generation": False,
    "features.memories": False,
    "features.shell_tool": False,
    "features.apps": False,
    "agents.enabled": False,
}

SUBAGENT_PROMPT = """You are a web research subagent. You receive one research task and \
return one report; nobody reads your messages until then and nobody can answer questions.

You work by writing programs for the code tool. The research tools are on its `tools` \
object; their results come back to your program, not to you: only what the program prints \
(`text(...)`, `inspect(...)`) reaches you. Let the program do the reading: run searches and \
fetches together, filter and cut the results in code, file the material, and print only \
the short excerpts you need to decide what to do next. Never print whole pages or long \
lists, and never copy material into your own messages to keep it: the files keep it.

Tools (each result is JSON in `result.text`: `JSON.parse(result.text)`):
- tools.research__deepseek_search({{ query, sites?, after?, before?, count? }}): DeepSeek's \
web search. Operators inside the query (site:, -site:, "exact phrase", -term, lang:zh,en, \
after:, before:), at most 128 characters in total; the task's time range is applied unless \
you pass after/before yourself. Returns {{ query, saved, results: [{{ n, title, url, site, \
date, summary, content }}] }}; `content` is the full text when DeepSeek has it, and `saved` \
the file it went to.
- tools.research__deepseek_fetch({{ url }}): one page as Markdown. Returns {{ title, url, \
chars, saved, content }}, the whole page in `content`.
- tools.research__tmp_write({{ path, content }}) / tmp_read({{ path, offset?, limit? }}) / \
tmp_list({{}}) / apply_patch({{ patch }}) (only when offered): files in your task \
directory, paths relative to it. Search results and pages are saved there already \
(search/NN.md, pages/NN-host.md). Write files the program builds with tmp_write; edit your \
notes with apply_patch (a Codex-style patch: *** Begin Patch / *** Add File: / \
*** Update File: / @@ hunks / *** End Patch).

Known limits of DeepSeek's index and fetcher:
- YouTube, Reddit, X/Twitter and Medium are neither indexed nor fetchable. Look for \
reposts, news coverage, mirrors or other sources instead, and say so in the report when \
the user asked for them.
- Zhihu and Hacker News are indexed but cannot be fetched: rely on search summaries and \
saved search content.
- Baidu Tieba threads (tieba.baidu.com/p/...) cannot be fetched, but search results carry \
the thread text: search with site:tieba.baidu.com and use their content.

Method:
1. One program per step, doing as much as it can: read the priority URLs first, if any; \
run several focused queries (Chinese and English where useful) at once with \
Promise.allSettled; fetch the most relevant pages the same way.
2. In the program, pick what matters: find the passages that answer the task (keywords, \
numbers, dates; a few hundred characters around each), and print them with their source.
3. File the material in code (when the file tools are offered): write materials.md with the passages that support your \
answer, each under its source title and URL, taken from the results by the program, not \
retyped by you. Prefer primary and recent sources; check facts that matter in a second \
source.
4. Stop when the task is answered or you are told time is up.

A program step looks like this:
```
const queries = ["...", "..."];
const found = await Promise.allSettled(
  queries.map((query) => tools.research__deepseek_search({{ query, count: 8 }})));
const hits = found.flatMap((r) => r.status === "fulfilled" ? JSON.parse(r.value.text).results : []);
const pages = await Promise.allSettled(hits.slice(0, 4).map((h) => tools.research__deepseek_fetch({{ url: h.url }})));
const excerpts = [];
for (const r of pages) {{
  if (r.status !== "fulfilled" || r.value.isError) continue;
  const page = JSON.parse(r.value.text);
  const at = page.content.search(/keyword/i);
  if (at >= 0) excerpts.push(`## ${{page.title}}\\n${{page.url}}\\n\\n${{page.content.slice(Math.max(0, at - 300), at + 900)}}`);
}}
await tools.research__tmp_write({{ path: "materials.md", content: excerpts.join("\\n\\n") }});
text(hits.map((h) => `${{h.n}}. ${{h.title}} ${{h.url}} ${{h.date ?? ""}}`).join("\\n"));
text(excerpts.map((e) => e.slice(0, 600)).join("\\n---\\n"));
```

Report (your final message):
- Language: the task instruction's language (Chinese when unclear).
- Answer the instruction directly, then the key findings, in your own words and short: \
the material is in the files, so do not restate it. Keep it under {report_chars} characters.
- Cite sources inline as [1], [2], ... and end with a source list: [n] title - URL (date).
- Only cite URLs you got from the tools. Say what you could not find or verify.
"""

WRAP_UP_NOTE = (
    "Time is almost up. Do not call any more tools; write the final report now "
    "from what you have."
)
REFUSED_NOTE = (
    "Not run: the research budget is used up. Write the final report now from what "
    "you have."
)


class DeepSeekError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class DeepSeekTools:
    """Client of DeepSeek's Search and Fetch APIs."""

    def __init__(self, api_key: str, base_url: str, proxy: str = "") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.proxy = proxy or None
        self._session: aiohttp.ClientSession | None = None

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def call(self, endpoint: str, body: JsonObject) -> JsonObject:
        """POST to ``/v1/tools/<endpoint>``, retrying what DeepSeek says to retry.

        Args:
            endpoint: ``search`` or ``fetch``.
            body: The JSON request body.

        Returns:
            The decoded JSON response.

        Raises:
            DeepSeekError: A non-200 answer that retrying did not fix.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=90)
            )
        url = f"{self.base_url}/v1/tools/{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(3):
            try:
                async with self._session.post(
                    url, json=body, headers=headers, proxy=self.proxy
                ) as resp:
                    text = await resp.text()
                    if resp.status == 200:
                        return json.loads(text)
                    try:
                        message = json.loads(text)["error"]["message"]
                    except (ValueError, KeyError, TypeError):
                        message = text[:300]
                    error = DeepSeekError(resp.status, message)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                error = DeepSeekError(0, f"network error: {e!s}")
            if error.status not in (0, *RETRY_STATUSES) or attempt == 2:
                raise error
            await asyncio.sleep(1.5 * (attempt + 1))
        raise AssertionError("unreachable")


class SandboxDir:
    """Read and write files under one directory of the chat's sandbox.

    The sandbox's file API only reaches ``/workspace``, so files go through
    its Python executor; the code is fixed and the request travels as base64,
    and the path is resolved inside the sandbox and refused outside ``base``.
    """

    MARK = "@@DSR@@"
    CODE = """
import base64 as _b64, json as _json, os as _os
def _dsr_op(req):
    base = _os.path.realpath(req["base"])
    _os.makedirs(base, exist_ok=True)
    path = _os.path.realpath(_os.path.join(base, req.get("path") or "."))
    if path != base and not path.startswith(base + _os.sep):
        return {"ok": False, "error": "path is outside the task directory"}
    op = req["op"]
    if op == "write":
        _os.makedirs(_os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(req["content"])
        return {"ok": True, "size": _os.path.getsize(path)}
    if op == "delete":
        if not _os.path.isfile(path):
            return {"ok": False, "error": "no such file"}
        _os.remove(path)
        return {"ok": True}
    if op == "read":
        if not _os.path.isfile(path):
            return {"ok": False, "error": "no such file"}
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        start = max(0, int(req.get("offset") or 0))
        return {"ok": True, "total": len(text),
                "text": text[start:start + int(req.get("limit") or 8000)]}
    files = []
    for root, _dirs, names in _os.walk(path):
        for name in names:
            p = _os.path.join(root, name)
            files.append([_os.path.relpath(p, base), _os.path.getsize(p)])
    return {"ok": True, "files": sorted(files)}
try:
    _dsr_out = _dsr_op(_json.loads(_b64.b64decode("%s").decode("utf-8")))
except Exception as _e:
    _dsr_out = {"ok": False, "error": repr(_e)}
print("@@DSR@@" + _json.dumps(_dsr_out, ensure_ascii=False))
del _dsr_op, _dsr_out
"""

    def __init__(self, get_booter, base: str) -> None:
        #: Coroutine function returning the chat's sandbox booter; called on
        #: first use so a task that saves nothing never boots a sandbox.
        self._get_booter = get_booter
        self._booter = None
        self.base = base

    async def op(self, op: str, path: str = ".", **fields: Any) -> JsonObject:
        if self._booter is None:
            self._booter = await self._get_booter()
        payload = json.dumps(
            {"op": op, "base": self.base, "path": path, **fields}, ensure_ascii=False
        )
        code = self.CODE % base64.b64encode(payload.encode("utf-8")).decode("ascii")
        result = await self._booter.python.exec(code, timeout=60)
        output = str(result.get("output") or "")
        for line in reversed(output.splitlines()):
            if line.startswith(self.MARK):
                return json.loads(line[len(self.MARK) :])
        error = str(result.get("error") or "") or output[-300:] or "no output"
        return {"ok": False, "error": f"sandbox: {error}"}


def parse_time_range(text: str, today: date) -> tuple[str, str]:
    """Turn a time range into DeepSeek's ``after:`` / ``before:`` values.

    Accepts ``7d`` / ``2w`` / ``3m`` / ``1y`` (that long until today), a date
    (``2025``, ``2025-06``, ``2025-06-01``; from then on) or ``A..B`` with
    either side left open.

    Args:
        text: The range as the caller wrote it; empty means none.
        today: The date relative ranges end on.

    Returns:
        ``(after, before)``, each "" when open.

    Raises:
        ValueError: The text is none of the forms above.
    """
    t = text.strip().lower().replace(" ", "")
    if not t:
        return "", ""
    if m := re.fullmatch(r"(\d+)([dwmy])", t):
        days = int(m.group(1)) * {"d": 1, "w": 7, "m": 30, "y": 365}[m.group(2)]
        return (today - timedelta(days=days)).isoformat(), ""
    ends = t.split("..", 1) if ".." in t else [t, ""]
    out = []
    for end in ends:
        if end and not re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", end):
            raise ValueError(f"bad date {end!r}")
        # DeepSeek documents years and full dates only.
        out.append(f"{end}-01" if len(end) == 7 else end)
    return out[0], out[1]


def _schema(props: JsonObject, required: list[str]) -> JsonObject:
    return {"type": "object", "properties": props, "required": required}


def dynamic_tools(with_files: bool) -> list[JsonObject]:
    """The subagent's tools, called from its program; each returns JSON text."""
    tools = [
        {
            "name": "deepseek_search",
            "description": (
                "Search the web with DeepSeek. Returns JSON { query, saved, results: "
                "[{ n, title, url, site, date, summary, content }] }; content is the "
                "full text when available, saved the file it went to."
            ),
            "inputSchema": _schema(
                {
                    "query": {
                        "type": "string",
                        "description": "Search terms, operators allowed.",
                    },
                    "sites": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only these sites (added as site: ... OR ...).",
                    },
                    "after": {
                        "type": "string",
                        "description": "YYYY or YYYY-MM-DD; omit for the task's "
                        "range, 'none' for no limit.",
                    },
                    "before": {
                        "type": "string",
                        "description": "YYYY or YYYY-MM-DD; omit for the task's "
                        "range, 'none' for no limit.",
                    },
                    "count": {"type": "integer", "description": "1-20, default 8."},
                },
                ["query"],
            ),
        },
        {
            "name": "deepseek_fetch",
            "description": (
                "Read a web page as Markdown. Returns JSON { title, url, chars, saved, "
                "content } with the whole page in content."
            ),
            "inputSchema": _schema({"url": {"type": "string"}}, ["url"]),
        },
    ]
    if with_files:
        tools += [
            {
                "name": "tmp_write",
                "description": "Write a file of the task directory (created or "
                "replaced). Returns JSON { path, chars }.",
                "inputSchema": _schema(
                    {
                        "path": {"type": "string", "description": "Relative path."},
                        "content": {"type": "string"},
                    },
                    ["path", "content"],
                ),
            },
            {
                "name": "tmp_read",
                "description": "Read a window of a file of the task directory. "
                "Returns JSON { path, total, text }.",
                "inputSchema": _schema(
                    {
                        "path": {"type": "string", "description": "Relative path."},
                        "offset": {"type": "integer"},
                        "limit": {
                            "type": "integer",
                            "description": f"Default 8000, max {MAX_READ_CHARS}.",
                        },
                    },
                    ["path"],
                ),
            },
            {
                "name": "apply_patch",
                "description": (
                    "Create, edit or delete files of the task directory with a "
                    "Codex-style patch (*** Begin Patch / *** Add File: / "
                    "*** Update File: / *** Move to: / *** Delete File: / @@ hunks / "
                    "*** End Patch). All hunks must apply or nothing is written. "
                    "Returns JSON { applied }."
                ),
                "inputSchema": _schema(
                    {"patch": {"type": "string", "description": "The full patch."}},
                    ["patch"],
                ),
            },
            {
                "name": "tmp_list",
                "description": "List the files of the task directory. Returns JSON "
                "{ files: [[path, size]] }.",
                "inputSchema": _schema({"path": {"type": "string"}}, []),
            },
        ]
    for tool in tools:
        tool["type"] = "function"
    return [
        {
            "type": "namespace",
            "name": NAMESPACE,
            "description": "Web research tools.",
            "tools": tools,
        }
    ]


def _text_result(text: str, success: bool = True) -> JsonObject:
    return {"contentItems": [{"type": "inputText", "text": text}], "success": success}


def _json_result(value: JsonObject) -> JsonObject:
    """A tool result for the subagent's program: JSON it parses."""
    return _text_result(json.dumps(value, ensure_ascii=False))


def _error_result(message: str) -> JsonObject:
    """A failed call, still JSON, so the program's JSON.parse never throws."""
    return _text_result(json.dumps({"error": message}, ensure_ascii=False), False)


class ResearchTask:
    """One research request run on its own restricted Codex thread."""

    def __init__(
        self,
        *,
        engine: Any,
        client: DeepSeekTools,
        sandbox: SandboxDir | None,
        task_id: str,
        model_provider: str | None,
        model: str | None,
        reasoning_effort: str,
        budget_s: float,
        max_tool_calls: int,
        search_count: int,
        report_chars: int,
        after: str,
        before: str,
        ori_question: str,
    ) -> None:
        self.engine = engine
        self.client = client
        self.sandbox = sandbox
        self.task_id = task_id
        self.model_provider = model_provider
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.budget_s = budget_s
        self.max_tool_calls = max_tool_calls
        self.search_count = search_count
        self.report_chars = report_chars
        self.after = after
        self.before = before
        self.ori_question = ori_question[:MAX_QUERY_CHARS]
        self.tool_calls = 0
        self.wrapping_up = False
        self.searches: list[str] = []
        self.pages: dict[str, JsonObject] = {}
        self.failed_fetches: list[str] = []
        self.saved_files = 0
        self.materials = False

    # ------------------------------------------------------------------ tools

    async def on_tool_call(self, msg: JsonObject) -> JsonObject:
        """Answer one dynamic tool call of the subagent."""
        name = str(msg.get("tool") or "")
        args = msg.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                return _error_result("arguments are not JSON")
        args = args if isinstance(args, dict) else {}
        if self.wrapping_up or self.tool_calls >= self.max_tool_calls:
            return _error_result(REFUSED_NOTE)
        self.tool_calls += 1
        try:
            if name == "deepseek_search":
                return await self._search(args)
            if name == "deepseek_fetch":
                return await self._fetch(args)
            if self.sandbox is not None and name in (
                "tmp_read",
                "tmp_list",
                "tmp_write",
                "apply_patch",
            ):
                return await self._files(name, args)
            return _error_result(f"unknown tool {name}")
        except DeepSeekError as e:
            if e.status == 402:
                return _error_result(
                    "the DeepSeek account has no balance; stop and report."
                )
            return _error_result(str(e))

    async def _save(self, path: str, content: str) -> str | None:
        """Saves to the sandbox; returns the path, or None when not saved."""
        if self.sandbox is None:
            return None
        try:
            res = await self.sandbox.op("write", path, content=content)
        except Exception as e:  # noqa: BLE001 - the research goes on without files
            logger.warning("deepseek_research: saving %s failed: %s", path, e)
            return None
        if not res.get("ok"):
            logger.warning("deepseek_research: saving %s failed: %s", path, res)
            return None
        self.saved_files += 1
        return path

    async def _search(self, args: JsonObject) -> JsonObject:
        parts = [str(args.get("query") or "").strip()]
        sites = [str(s).strip() for s in args.get("sites") or [] if str(s).strip()]
        if sites:
            parts.append(" OR ".join(f"site:{s}" for s in sites))
        for op, default in (("after", self.after), ("before", self.before)):
            value = str(args.get(op) if args.get(op) is not None else default).strip()
            if value and value.lower() != "none" and f"{op}:" not in parts[0]:
                parts.append(f"{op}:{value}")
        query = " ".join(p for p in parts if p)
        if not parts[0]:
            return _error_result("empty query")
        if len(query) > MAX_QUERY_CHARS:
            return _error_result(
                f"the query is {len(query)} characters with operators; "
                f"DeepSeek allows {MAX_QUERY_CHARS}. Shorten it."
            )
        try:
            count = max(1, min(20, int(args.get("count") or self.search_count)))
        except (TypeError, ValueError):
            count = self.search_count
        body: JsonObject = {"query": query, "count": count, "need_content": True}
        if self.ori_question:
            body["ori_question"] = self.ori_question
        results = (await self.client.call("search", body)).get("results") or []
        self.searches.append(query)
        out: list[JsonObject] = []
        saved: list[str] = []
        for i, r in enumerate(results, 1):
            content = str(r.get("content") or "")[:MAX_PAGE_CHARS]
            out.append(
                {
                    "n": i,
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "site": r.get("site_name") or "",
                    "date": r.get("date") or "",
                    "summary": r.get("summary") or r.get("snippet") or "",
                    "content": content,
                }
            )
            if content:
                saved.append(
                    f"## [{i}] {r.get('title') or ''}\n{r.get('url') or ''}\n\n{content}\n"
                )
        path = None
        if saved:
            path = await self._save(
                f"search/{len(self.searches):02d}.md",
                f"# {query}\n\n" + "\n".join(saved),
            )
        return _json_result({"query": query, "saved": path, "results": out})

    async def _fetch(self, args: JsonObject) -> JsonObject:
        url = str(args.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            return _error_result("url must be http(s)")
        page = self.pages.get(url)
        if page is None:
            try:
                data = await self.client.call("fetch", {"url": url})
            except DeepSeekError as e:
                if e.status in (403, 404):
                    self.failed_fetches.append(url)
                    return _error_result(
                        f"DeepSeek could not fetch this page ({e!s}): a site it does "
                        "not reach, anti-bot protection, or no readable content. Use "
                        "search results or another source."
                    )
                raise
            content = str(data.get("content") or "")[:MAX_PAGE_CHARS]
            host = re.sub(r"[^A-Za-z0-9.-]", "_", urlparse(url).hostname or "page")
            path = f"pages/{len(self.pages) + 1:02d}-{host}.md"
            page = {"title": str(data.get("title") or ""), "content": content}
            self.pages[url] = page
            page["saved"] = await self._save(
                path, f"# {page['title']}\n{url}\n\n{content}"
            )
        return _json_result(
            {
                "title": page["title"],
                "url": url,
                "chars": len(page["content"]),
                "saved": page["saved"],
                "content": page["content"],
            }
        )

    async def _files(self, name: str, args: JsonObject) -> JsonObject:
        path = str(args.get("path") or ".")
        try:
            if name == "tmp_read":
                limit = max(1, min(MAX_READ_CHARS, int(args.get("limit") or 8000)))
                res = await self.sandbox.op(
                    "read", path, offset=int(args.get("offset") or 0), limit=limit
                )
                if res.get("ok"):
                    return _json_result(
                        {"path": path, "total": res["total"], "text": res["text"]}
                    )
            elif name == "apply_patch":
                return await self._apply_patch(str(args.get("patch") or ""))
            elif name == "tmp_write":
                content = str(args.get("content") or "")
                if len(content) > MAX_WRITE_CHARS:
                    return _error_result(
                        f"content is {len(content)} characters; at most "
                        f"{MAX_WRITE_CHARS} per file"
                    )
                res = await self.sandbox.op("write", path, content=content)
                if res.get("ok"):
                    self.saved_files += 1
                    self.materials = self.materials or path == "materials.md"
                    return _json_result({"path": path, "chars": len(content)})
            else:
                res = await self.sandbox.op("list", path)
                if res.get("ok"):
                    return _json_result({"files": res["files"]})
        except (TypeError, ValueError) as e:
            return _error_result(str(e))
        except Exception as e:  # noqa: BLE001 - a sandbox failure is the tool's result
            return _error_result(f"sandbox unavailable: {e!s}")
        return _error_result(str(res.get("error")))

    async def _apply_patch(self, patch: str) -> JsonObject:
        """Apply a Codex-style patch to the task directory, all or nothing.

        Every file's new content is worked out first (later operations on a
        file see the earlier ones), and only then written, so a hunk that does
        not apply leaves every file as it was.
        """
        try:
            ops = parse_patch(patch)
        except PatchError as e:
            return _error_result(f"invalid patch: {e}")
        # Refused before anything is written; the sandbox checks again after
        # resolving links.
        base = self.sandbox.base.rstrip("/") + "/"
        for path in [p for op in ops for p in (op.path, op.move_to) if p]:
            if ".." in path.split("/") or (
                path.startswith("/") and not path.startswith(base)
            ):
                return _error_result(
                    f"{path}: path is outside the task directory; nothing written"
                )
        # path -> new content, or None to delete it; in patch order.
        pending: dict[str, str | None] = {}
        done: list[str] = []
        for op in ops:
            if op.kind == "add":
                pending[op.path] = "\n".join(op.add_lines) + "\n"
                done.append(f"A {op.path}")
                continue
            if op.kind == "delete":
                pending[op.path] = None
                done.append(f"D {op.path}")
                continue
            if op.path in pending:
                content = pending[op.path]
                if content is None:
                    return _error_result(f"{op.path} is deleted above")
            else:
                res = await self.sandbox.op(
                    "read", op.path, limit=MAX_PATCH_FILE_CHARS + 1
                )
                if not res.get("ok"):
                    return _error_result(f"cannot read {op.path}: {res.get('error')}")
                if res["total"] > MAX_PATCH_FILE_CHARS:
                    return _error_result(f"{op.path} is too large to patch")
                content = res["text"]
            try:
                new = apply_hunks(content, op.hunks)
            except PatchError as e:
                return _error_result(f"{op.path}: {e}; nothing written")
            target = op.move_to or op.path
            if target != op.path:
                pending[op.path] = None
            pending[target] = new
            done.append(f"M {op.path}" + (f" -> {target}" if target != op.path else ""))
        written: list[str] = []
        for path, content in pending.items():
            res = await self.sandbox.op(
                "write" if content is not None else "delete",
                path,
                **({"content": content} if content is not None else {}),
            )
            if not res.get("ok"):
                note = f" (already applied: {', '.join(written)})" if written else ""
                return _error_result(f"{path}: {res.get('error')}{note}")
            written.append(path)
        self.saved_files += 1
        return _json_result({"applied": done})

    # -------------------------------------------------------------------- run

    async def run(self, task_text: str) -> str:
        """Run the subagent to its report, within the time budget.

        Args:
            task_text: The task as the subagent's first and only input.

        Returns:
            The report, or what is left of the research when it failed or ran
            out of time, followed by a line on the task's files.
        """
        started = time.monotonic()
        params: JsonObject = {
            "cwd": "/",
            "ephemeral": True,
            "no_environment": True,
            "base_instructions": SUBAGENT_PROMPT.format(
                report_chars=int(self.report_chars * 0.8)
            ),
            "dynamic_tools": dynamic_tools(self.sandbox is not None),
            "config": dict(THREAD_CONFIG),
        }
        if self.model_provider:
            params["model_provider"] = self.model_provider
        if self.model:
            params["model"] = self.model
        if self.reasoning_effort:
            params["config"]["model_reasoning_effort"] = self.reasoning_effort
        info = json.loads(
            await self.engine.rt.start_thread(json.dumps(params, ensure_ascii=False))
        )
        thread_id = info["thread_id"]
        pump = self.engine.pump(thread_id)
        queue = pump.open_turn(self.on_tool_call)
        texts: list[str] = []
        errors: list[str] = []
        finished = timed_out = False
        try:
            sub = await self.engine.submit_turn(
                thread_id,
                {
                    "input": [{"type": "text", "text": task_text, "text_elements": []}],
                    "mode": "start_if_idle",
                },
            )
            if sub.get("status") != "started":
                return f"调研没能启动：{sub}"
            turn_id = sub.get("turn_id")
            loop = asyncio.get_running_loop()
            soft = loop.time() + self.budget_s * WRAP_UP_AT
            hard = loop.time() + self.budget_s
            while True:
                if not self.wrapping_up and loop.time() >= soft:
                    self.wrapping_up = True
                    with contextlib.suppress(Exception):
                        await self.engine.submit_turn(
                            thread_id,
                            {
                                "input": [
                                    {
                                        "type": "text",
                                        "text": WRAP_UP_NOTE,
                                        "text_elements": [],
                                    }
                                ],
                                "mode": "steer",
                                "expected_turn_id": turn_id,
                            },
                        )
                wait = (hard if self.wrapping_up else soft) - loop.time()
                try:
                    msg = await asyncio.wait_for(queue.get(), max(0.05, wait))
                except asyncio.TimeoutError:
                    if self.wrapping_up:
                        timed_out = True
                        break
                    continue
                kind = msg.get("type")
                if kind == "agent_message" and msg.get("phase") in (
                    None,
                    "final_answer",
                ):
                    texts.append(str(msg.get("message") or ""))
                elif kind == "error":
                    errors.append(str(msg.get("message") or ""))
                elif kind in TERMINAL_EVENTS or kind == "_pump_closed":
                    finished = kind != "_pump_closed"
                    break
        finally:
            if not finished:
                with contextlib.suppress(Exception):
                    await self.engine.interrupt(thread_id)
            pump.close_turn()
            with contextlib.suppress(Exception):
                await self.engine.forget_thread(thread_id)
        report = next((t.strip() for t in reversed(texts) if t.strip()), "")
        report_saved = bool(report) and bool(await self._save("report.md", report))
        elapsed = time.monotonic() - started
        logger.info(
            "deepseek_research %s: %.0fs, %d tool calls, %d searches, %d pages, "
            "finished=%s timed_out=%s",
            self.task_id,
            elapsed,
            self.tool_calls,
            len(self.searches),
            len(self.pages),
            finished,
            timed_out,
        )
        if not report:
            reason = (
                "超出时间预算"
                if timed_out
                else (errors[-1] if errors else "没有产出报告")
            )
            lines = [f"调研未完成（{reason}）。"]
            if self.searches:
                lines.append("已执行的搜索：" + "；".join(self.searches))
            if self.pages:
                lines.append(
                    "已读取的网页：\n"
                    + "\n".join(f"- {p['title']} {u}" for u, p in self.pages.items())
                )
            report = "\n".join(lines)
        elif timed_out:
            report += "\n\n（调研超时，以上为部分结果。）"
        if len(report) > self.report_chars:
            report = report[: self.report_chars] + "\n…（报告过长已截断）"
        footer = (
            f"用时 {elapsed:.0f} 秒，搜索 {len(self.searches)} 次，"
            f"读取网页 {len(self.pages)} 个"
        )
        if self.failed_fetches:
            footer += f"，抓取失败 {len(self.failed_fetches)} 个"
        if self.sandbox is not None and self.saved_files:
            parts = ["report.md 为完整报告"] if report_saved else []
            if self.materials:
                parts.append("materials.md 为子代理整理的资料摘录")
            if self.searches:
                parts.append("search/ 为搜索结果正文")
            if self.pages:
                parts.append("pages/ 为网页")
            footer += f"。原始材料在沙箱 {self.sandbox.base}/（{'，'.join(parts)}）"
        return f"{report}\n\n---\n{footer}。"


def new_task_id() -> str:
    return uuid.uuid4().hex[:12]
