"""Unit tests for model_select (no AstrBot needed):
``python -m pytest tests/test_model_select.py`` from the plugin directory."""

import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "model_select", pathlib.Path(__file__).resolve().parent.parent / "model_select.py"
)
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)

DEEPSEEK = {
    "id": "deepseek",
    "base_url": "https://api.deepseek.com/v1",
    "models": [{"slug": "deepseek-flash"}, {"slug": "deepseek-pro"}],
}
LOCAL = {
    "id": "halogen-395",
    "name": "395 halogen",
    "base_url": "http://192.168.6.116:8731/v1",
    "models": [{"slug": "halogen-qwen3.8-flash-next"}],
}
OTHER = {"id": "commandcode", "base_url": "https://api.commandcode.ai/provider/v1"}
RUNNER = {
    "model_providers": [OTHER, DEEPSEEK, LOCAL],
    "model_provider": "commandcode",
    "model": "deepseek/deepseek-v4.1-flash",
}


def pick(config, runner=RUNNER):
    return ms.pick_subagent_model(config, runner)


def test_auto_default_takes_deepseek_provider():
    assert pick({}) == ("deepseek", "deepseek-flash")
    assert pick({"subagent_model_source": "auto"}) == ("deepseek", "deepseek-flash")


def test_auto_named_provider_and_model():
    cfg = {"subagent_provider": "halogen-395"}
    assert pick(cfg) == ("halogen-395", "halogen-qwen3.8-flash-next")
    cfg = {"subagent_provider": "395 halogen", "subagent_model": "x"}
    assert pick(cfg) == ("halogen-395", "x")


def test_auto_without_deepseek_falls_back_to_runner():
    runner = {**RUNNER, "model_providers": [OTHER]}
    assert pick({}, runner) == ("commandcode", "deepseek/deepseek-v4.1-flash")


def test_auto_unknown_provider_is_an_error():
    with pytest.raises(ValueError, match="不在"):
        pick({"subagent_provider": "nope"})


def test_deepseek_source():
    cfg = {"subagent_model_source": "deepseek", "subagent_model": "deepseek-pro"}
    assert pick(cfg) == ("deepseek", "deepseek-pro")
    # subagent_provider is ignored outside auto.
    cfg = {"subagent_model_source": "deepseek", "subagent_provider": "halogen-395"}
    assert pick(cfg) == ("deepseek", "deepseek-flash")
    with pytest.raises(ValueError, match="api.deepseek.com"):
        pick(
            {"subagent_model_source": "deepseek"},
            {**RUNNER, "model_providers": [LOCAL]},
        )


def test_local_source():
    cfg = {"subagent_model_source": "local", "local_provider": "halogen-395"}
    assert pick(cfg) == ("halogen-395", "halogen-qwen3.8-flash-next")
    # subagent_model belongs to the DeepSeek side; local_model wins here.
    cfg.update(subagent_model="deepseek-pro", local_model="other-slug")
    assert pick(cfg) == ("halogen-395", "other-slug")


def test_local_source_errors():
    with pytest.raises(ValueError, match="local_provider"):
        pick({"subagent_model_source": "local"})
    with pytest.raises(ValueError, match="不在"):
        pick({"subagent_model_source": "local", "local_provider": "nope"})
    bare = {**LOCAL, "models": []}
    with pytest.raises(ValueError, match="local_model"):
        pick(
            {"subagent_model_source": "local", "local_provider": "halogen-395"},
            {**RUNNER, "model_providers": [bare]},
        )


def test_invalid_source():
    with pytest.raises(ValueError, match="subagent_model_source"):
        pick({"subagent_model_source": "cloud"})


def test_source_is_case_insensitive():
    cfg = {"subagent_model_source": " Local ", "local_provider": "halogen-395"}
    assert pick(cfg)[0] == "halogen-395"
