"""AstrBotX plugin: web research by a restricted Codex subagent on DeepSeek's
Search and Fetch APIs."""

from __future__ import annotations

import re
from datetime import date

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

from .research import (
    SANDBOX_ROOT,
    DeepSeekTools,
    ResearchTask,
    SandboxDir,
    new_task_id,
    parse_time_range,
)

# Seconds the subagent leaves before AstrBot's tool_call_timeout cancels it.
TIMEOUT_MARGIN_S = 15
MIN_BUDGET_S = 30


def _provider_id(raw: object) -> str:
    # As the Codex runner names WebUI providers (model_provider_overrides).
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(raw or "").strip())


@register(
    "deepseek_research",
    "xkeyC",
    "Web research by a restricted Codex subagent on DeepSeek Search & Fetch",
    "0.1.0",
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

    async def terminate(self) -> None:
        await self.client.close()

    def _subagent_model(self, runner_cfg: dict) -> tuple[str | None, str | None]:
        """The Codex provider and model the subagent runs on.

        The plugin setting names one of the runner's WebUI providers; unset,
        the first provider on api.deepseek.com is taken, else the runner's own
        provider and model.

        Returns:
            ``(provider id, model)``; None leaves Codex's default.

        Raises:
            ValueError: The setting names no provider, or a provider with no
                model to use.
        """
        providers = [
            p for p in runner_cfg.get("model_providers") or [] if isinstance(p, dict)
        ]
        wanted = _provider_id(self.config.get("subagent_provider"))
        model = str(self.config.get("subagent_model") or "").strip()
        if wanted:
            match = next(
                (
                    p
                    for p in providers
                    if wanted in (_provider_id(p.get("id")), str(p.get("name") or ""))
                ),
                None,
            )
            if match is None:
                if wanted != _provider_id(runner_cfg.get("model_provider")):
                    raise ValueError(
                        f"子代理 provider {wanted!r} 不在 Codex runner 的 providers 里"
                    )
                return wanted, model or str(runner_cfg.get("model") or "") or None
        else:
            match = next(
                (p for p in providers if "api.deepseek.com" in str(p.get("base_url"))),
                None,
            )
            if match is None:
                return (
                    str(runner_cfg.get("model_provider") or "") or None,
                    model or str(runner_cfg.get("model") or "") or None,
                )
        if not model:
            slugs = [
                str(m.get("slug") or "").strip()
                for m in match.get("models") or []
                if isinstance(m, dict)
            ]
            model = next((s for s in slugs if s), "")
        if not model:
            raise ValueError(
                f"provider {match.get('id')!r} 没有列出模型，请在插件配置里填 subagent_model"
            )
        return _provider_id(match.get("id")), model

    @filter.llm_tool(name="deepseek_research")
    async def deepseek_research(
        self,
        event: AstrMessageEvent,
        keywords: str,
        instruction: str = "",
        time_range: str = "",
        priority_urls: list[str] | None = None,
    ) -> str:
        """Research a topic on the web and return a cited report. A subagent searches and reads pages with DeepSeek's search engine (good Chinese coverage, incl. Tieba, Bilibili, Weibo, Zhihu summaries) and writes the report; its raw material stays in the sandbox under /tmp/ds_research/. It cannot reach YouTube, Reddit, X/Twitter or Medium. Takes tens of seconds to minutes.

        Args:
            keywords(string): What to research: keywords or a short topic.
            instruction(string): What the report should answer or contain, its focus and format.
            time_range(string): Optional. Publication time to search: 7d / 2w / 3m / 1y (until today), a date such as 2025 or 2025-06-01 (from then on), or A..B such as 2025-01-01..2025-06-30.
            priority_urls(array[string]): Optional. URLs the subagent reads first.
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
        try:
            provider, model = self._subagent_model(runner_cfg)
        except ValueError as e:
            return f"deepseek_research 配置有误：{e}"
        # The tool runs inside AstrBot's tool_call_timeout; past it the call is
        # cancelled and the report lost, so the subagent wraps up before.
        timeout = int(runner_cfg.get("tool_call_timeout") or 120)
        budget = min(
            float(self.config.get("time_budget_s") or 600),
            float(timeout - TIMEOUT_MARGIN_S),
        )
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
            reasoning_effort=str(self.config.get("subagent_reasoning_effort") or ""),
            budget_s=budget,
            max_tool_calls=int(self.config.get("max_tool_calls") or 30),
            search_count=int(self.config.get("search_count") or 8),
            fetch_window=int(self.config.get("fetch_window_chars") or 6000),
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
            "deepseek_research %s for %s: provider=%s model=%s budget=%ss sandbox=%s",
            task_id,
            umo,
            provider,
            model,
            int(budget),
            sandbox is not None,
        )
        try:
            return await task.run("\n".join(lines))
        except Exception as e:  # noqa: BLE001 - the main agent gets the reason
            logger.error("deepseek_research %s failed: %s", task_id, e, exc_info=True)
            return f"deepseek_research 失败：{e!s}"
