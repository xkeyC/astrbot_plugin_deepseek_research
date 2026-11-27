"""AstrBotX plugin: web research by a restricted Codex subagent on DeepSeek's
Search and Fetch APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from astrbot.api import FunctionTool, logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Context, Star, register

from .model_select import pick_subagent_model
from .research import (
    SANDBOX_ROOT,
    DeepSeekTools,
    ResearchTask,
    SandboxDir,
    new_task_id,
    parse_time_range,
)

# Seconds the subagent leaves before a tool_call_timeout that cancels it.
TIMEOUT_MARGIN_S = 15
MIN_BUDGET_S = 30


@dataclass
class DeepSeekResearchTool(FunctionTool):
    """``deepseek_research``. A FunctionTool rather than an ``llm_tool``
    handler: only ``call`` gets the run context, which says whether this run
    cancels a tool at its timeout."""

    name: str = "deepseek_research"
    description: str = (
        "Fallback for Chinese-language information that openai_research could "
        "not find (Chinese sites and news, Tieba, Bilibili, Weibo, Zhihu "
        "summaries): use openai_research first, for Chinese topics too; it is "
        "faster. Returns a cited report. A subagent searches and reads pages with DeepSeek's "
        "search engine and writes the report; its raw material stays in the "
        "sandbox under /tmp/ds_research/. It cannot reach YouTube, Reddit, "
        "X/Twitter or Medium. Takes one to several minutes."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "What to research: keywords or a short topic.",
                },
                "instruction": {
                    "type": "string",
                    "description": "What the report should answer or contain, its "
                    "focus and format.",
                },
                "time_range": {
                    "type": "string",
                    "description": "Optional. Publication time to search: 7d / 2w / "
                    "3m / 1y or day / week / month / year (until today), a date such as 2025 or 2025-06-01 (from "
                    "then on), or A..B such as 2025-01-01..2025-06-30.",
                },
                "priority_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional. URLs the subagent reads first.",
                },
            },
            "required": ["keywords"],
        }
    )
    plugin: Any = None

    async def call(self, context: Any, **kwargs: Any) -> str:
        # Runs that offer astrbot_tool_task keep a tool past its timeout and
        # let the model decide; every other run (an older AstrBot, a plugin's
        # tool_loop_agent, a background task) cancels it there.
        timeout = None
        if getattr(context, "tool_tasks", False) is not True:
            timeout = int(getattr(context, "tool_call_timeout", 0) or 120)
        return await self.plugin.research(
            context.context.event,
            cancel_after_s=timeout,
            keywords=str(kwargs.get("keywords") or ""),
            instruction=str(kwargs.get("instruction") or ""),
            time_range=str(kwargs.get("time_range") or ""),
            priority_urls=kwargs.get("priority_urls"),
        )


@register(
    "deepseek_research",
    "xkeyC",
    "Web research by a restricted Codex subagent on DeepSeek Search & Fetch",
    "0.3.0",
)
class DeepSeekResearchPlugin(Star):
    def __init__(self, context: Context, config: dict) -> None:
        super().__init__(context)
        self.config = config
        self.client = DeepSeekTools(
            str(config.get("deepseek_api_key") or "").strip(),
            str(config.get("deepseek_base_url") or "https://api.deepseek.com"),
            str(config.get("proxy") or "").strip(),
        )
        self.context.add_llm_tools(DeepSeekResearchTool(plugin=self))

    async def terminate(self) -> None:
        await self.client.close()

    async def research(
        self,
        event: AstrMessageEvent,
        *,
        cancel_after_s: int | None,
        keywords: str,
        instruction: str = "",
        time_range: str = "",
        priority_urls: list[str] | None = None,
    ) -> str:
        """Run one research task for ``deepseek_research``.

        Args:
            event: The message event the tool call belongs to.
            cancel_after_s: The tool_call_timeout that cancels this call, or
                None when the run keeps it going past that (tool tasks).
            keywords: What to research.
            instruction: What the report should answer.
            time_range: The publication time range, see ``parse_time_range``.
            priority_urls: URLs to read first.

        Returns:
            The report, or why there is none.
        """
        if not self.client.api_key:
            return "deepseek_research 未配置 DeepSeek API key。"
        try:
            after, before = parse_time_range(time_range or "", date.today())
        except ValueError as e:
            return (
                f"time_range 无法识别（{e}）：请用 7d / 2w / 3m / 1y、2025-06-01 "
                "或 2025-01-01..2025-06-30。"
            )

        from astrbot.core.agent.runners.codex.codex_agent_runner import engine_options
        from astrbot.core.agent.runners.codex.native import CodexEngine
        from astrbot.core.computer.computer_client import get_booter
        from astrbot.core.config.agent_runner import normalize_agent_runner

        umo = event.unified_msg_origin
        chat_cfg = self.context.get_config(umo=umo)
        runner = normalize_agent_runner(chat_cfg.get("agent_runner"))
        runner_cfg = runner["config"]
        # The chat's own model; DeepSeek is only the search engine.
        provider, model, effort = pick_subagent_model(self.config, runner_cfg)
        budget = float(self.config.get("time_budget_s") or 600)
        if cancel_after_s is not None:
            # Cancelled there, the report would be lost: wrap up before.
            budget = min(budget, float(cancel_after_s - TIMEOUT_MARGIN_S))
        budget = max(float(MIN_BUDGET_S), budget)

        task_id = new_task_id()
        sandbox = None
        if (
            chat_cfg.get("provider_settings", {}).get("computer_use_runtime")
            == "sandbox"
        ):
            sandbox = SandboxDir(
                lambda: get_booter(self.context, umo), f"{SANDBOX_ROOT}/{task_id}"
            )
        task = ResearchTask(
            # The chat's own engine: other options would restart it.
            engine=await CodexEngine.get(engine_options(runner_cfg)),
            client=self.client,
            sandbox=sandbox,
            task_id=task_id,
            model_provider=provider,
            model=model,
            reasoning_effort=effort,
            budget_s=budget,
            max_tool_calls=int(self.config.get("max_tool_calls") or 30),
            search_count=int(self.config.get("search_count") or 8),
            report_chars=int(self.config.get("result_max_chars") or 8000),
            after=after,
            before=before,
            ori_question=instruction or keywords,
        )
        urls = [str(u).strip() for u in priority_urls or [] if str(u).strip()]
        lines = [
            f"Today is {date.today().isoformat()}.",
            f"Topic / keywords: {keywords}",
            f"Instruction: {instruction or '(none: give a well-sourced overview)'}",
        ]
        if after or before:
            lines.append(
                f"Time range: {after or 'open'} .. {before or 'open'} (applied to searches)"
            )
        if urls:
            lines.append("Read these first:\n" + "\n".join(f"- {u}" for u in urls))
        lines.append(
            f"Budget: about {int(budget)} seconds and {task.max_tool_calls} tool calls."
        )
        logger.info(
            "deepseek_research %s for %s: provider=%s model=%s effort=%s budget=%ss sandbox=%s",
            task_id,
            umo,
            provider,
            model,
            effort or "default",
            int(budget),
            sandbox is not None,
        )
        try:
            return await task.run("\n".join(lines))
        except Exception as e:  # noqa: BLE001 - the main agent gets the reason
            logger.error("deepseek_research %s failed: %s", task_id, e, exc_info=True)
            return f"deepseek_research 失败：{e!s}"
