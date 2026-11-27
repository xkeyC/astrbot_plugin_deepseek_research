"""Which Codex provider and model the research subagent runs on.

Kept free of AstrBot imports so it can be tested on its own.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# subagent_model_source values.
SOURCE_AUTO = "auto"
SOURCE_DEEPSEEK = "deepseek"
SOURCE_LOCAL = "local"
SOURCES = (SOURCE_AUTO, SOURCE_DEEPSEEK, SOURCE_LOCAL)

DEEPSEEK_HOST = "api.deepseek.com"


def provider_id(raw: object) -> str:
    """A WebUI provider's ID as the Codex runner names it
    (model_provider_overrides)."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(raw or "").strip())


def _first_model(provider: Mapping[str, Any]) -> str:
    for m in provider.get("models") or []:
        if isinstance(m, Mapping) and str(m.get("slug") or "").strip():
            return str(m["slug"]).strip()
    return ""


def _find(providers: list[Mapping[str, Any]], raw: object) -> Mapping[str, Any] | None:
    """The provider a setting names: by ID (normalised as the runner does) or
    by its WebUI name as written."""
    name = str(raw or "").strip()
    wanted = provider_id(name)
    return next(
        (
            p
            for p in providers
            if provider_id(p.get("id")) == wanted
            or str(p.get("name") or "").strip() == name
        ),
        None,
    )


def _with_model(
    provider: Mapping[str, Any], model: str, setting: str
) -> tuple[str, str]:
    model = model or _first_model(provider)
    if not model:
        raise ValueError(
            f"provider {provider.get('id')!r} 没有列出模型，请在插件配置里填 {setting}"
        )
    return provider_id(provider.get("id")), model


def pick_subagent_model(
    config: Mapping[str, Any], runner_cfg: Mapping[str, Any]
) -> tuple[str | None, str | None]:
    """The provider and model for the subagent.

    ``subagent_model_source``:

    * ``auto`` (default, the behaviour before the setting existed):
      ``subagent_provider`` if set; else the first provider on
      api.deepseek.com; else the runner's own provider and model.
    * ``deepseek``: the first provider on api.deepseek.com, model
      ``subagent_model``.
    * ``local``: the provider named by ``local_provider`` (e.g. a local
      halogen server), model ``local_model``.

    An unset model is the provider's first listed one.

    Args:
        config: The plugin configuration.
        runner_cfg: The chat's Codex runner configuration.

    Returns:
        ``(provider id, model)``; None leaves Codex's default.

    Raises:
        ValueError: The settings name no usable provider or model.
    """
    providers = [
        p for p in runner_cfg.get("model_providers") or [] if isinstance(p, Mapping)
    ]
    source = str(config.get("subagent_model_source") or SOURCE_AUTO).strip().lower()
    if source not in SOURCES:
        raise ValueError(
            f"subagent_model_source {source!r} 无效，可选 {' / '.join(SOURCES)}"
        )
    model = str(config.get("subagent_model") or "").strip()

    if source == SOURCE_LOCAL:
        wanted = provider_id(config.get("local_provider"))
        if not wanted:
            raise ValueError("选了本地模型（local），但没填 local_provider")
        match = _find(providers, config.get("local_provider"))
        if match is None:
            raise ValueError(
                f"本地模型 provider {wanted!r} 不在 Codex runner 的 providers 里"
            )
        return _with_model(
            match, str(config.get("local_model") or "").strip(), "local_model"
        )

    deepseek = next(
        (p for p in providers if DEEPSEEK_HOST in str(p.get("base_url") or "")), None
    )
    if source == SOURCE_DEEPSEEK:
        if deepseek is None:
            raise ValueError(
                f"选了 DeepSeek 模型，但没有 base_url 指向 {DEEPSEEK_HOST} 的 provider"
            )
        return _with_model(deepseek, model, "subagent_model")

    # auto
    wanted = provider_id(config.get("subagent_provider"))
    if wanted:
        match = _find(providers, config.get("subagent_provider"))
        if match is None:
            if wanted != provider_id(runner_cfg.get("model_provider")):
                raise ValueError(
                    f"子代理 provider {wanted!r} 不在 Codex runner 的 providers 里"
                )
            return wanted, model or str(runner_cfg.get("model") or "") or None
        return _with_model(match, model, "subagent_model")
    if deepseek is not None:
        return _with_model(deepseek, model, "subagent_model")
    return (
        str(runner_cfg.get("model_provider") or "") or None,
        model or str(runner_cfg.get("model") or "") or None,
    )
