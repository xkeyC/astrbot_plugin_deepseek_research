"""Which Codex provider, model and reasoning effort the research subagent
runs on: the chat's own Codex runner's. DeepSeek is only the search engine
(its Search / Fetch APIs), never the model.

Kept free of AstrBot imports so it can be tested on its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def pick_subagent_model(
    config: Mapping[str, Any], runner_cfg: Mapping[str, Any]
) -> tuple[str | None, str | None, str]:
    """The provider, model and reasoning effort for the subagent.

    Args:
        config: The plugin configuration; ``subagent_reasoning_effort``
            overrides the runner's effort when set.
        runner_cfg: The chat's Codex runner configuration.

    Returns:
        ``(provider id, model, effort)``; None / "" leave Codex's default.
    """
    provider = str(runner_cfg.get("model_provider") or "").strip() or None
    model = str(runner_cfg.get("model") or "").strip() or None
    effort = (
        str(config.get("subagent_reasoning_effort") or "").strip()
        or str(runner_cfg.get("reasoning_effort") or "").strip()
    )
    return provider, model, effort
