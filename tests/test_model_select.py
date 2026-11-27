"""Unit tests for model_select (no AstrBot needed):
``python -m pytest tests/test_model_select.py`` from the plugin directory."""

import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "model_select", pathlib.Path(__file__).resolve().parent.parent / "model_select.py"
)
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)

DEEPSEEK = {
    "id": "deepseek",
    "base_url": "https://api.deepseek.com/v1",
    "models": [{"slug": "deepseek-flash"}],
}
LOCAL = {
    "id": "halogen-395",
    "base_url": "http://192.168.6.116:8731/v1",
    "models": [{"slug": "halogen-qwen3.8-flash-next"}],
}
RUNNER = {
    "model_providers": [DEEPSEEK, LOCAL],
    "model_provider": "halogen-395",
    "model": "halogen-qwen3.8-flash-next",
    "reasoning_effort": "medium",
}


def test_runs_on_the_runner_model_even_with_a_deepseek_provider():
    assert ms.pick_subagent_model({}, RUNNER) == (
        "halogen-395",
        "halogen-qwen3.8-flash-next",
        "medium",
    )


def test_old_model_settings_are_ignored():
    old = {
        "subagent_model_source": "deepseek",
        "subagent_provider": "deepseek",
        "subagent_model": "deepseek-flash",
        "local_provider": "other",
    }
    assert ms.pick_subagent_model(old, RUNNER)[:2] == (
        "halogen-395",
        "halogen-qwen3.8-flash-next",
    )


def test_effort_override():
    assert (
        ms.pick_subagent_model({"subagent_reasoning_effort": " low "}, RUNNER)[2]
        == "low"
    )


def test_empty_runner_leaves_codex_defaults():
    assert ms.pick_subagent_model({}, {}) == (None, None, "")
