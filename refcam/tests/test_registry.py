import pytest

from refcam.__main__ import build_parser
from refcam.core.config import load_config
from refcam.core.registry import Registry, load_commands, load_detectors


def test_all_commands_are_discovered():
    names = set(load_commands())
    assert {"run", "add", "list", "remove", "overlay", "test"} <= names


def test_parser_builds_every_command():
    parser = build_parser()
    subparsers = [
        action for action in parser._actions if action.dest == "command"
    ][0]
    assert {"run", "add", "list", "remove", "overlay", "test"} <= set(
        subparsers.choices
    )


def test_sift_detector_is_discovered():
    assert "sift" in load_detectors()


def test_default_config_loads():
    config = load_config()
    assert config["resolution_width"] == 640
    assert config["resolution_height"] == 480
    assert config["match_ratio"] == 0.75


def test_duplicate_registration_rejected():
    registry = Registry("thing", "refcam.detectors")

    @registry.register("x")
    def item():
        return None

    with pytest.raises(ValueError, match="already registered"):
        registry.register("x")(object())


def test_unknown_name_lists_available_names():
    registry = Registry("thing", "refcam.detectors")
    registry.register("a")(object())
    with pytest.raises(KeyError, match="available: a"):
        registry.get("missing")
