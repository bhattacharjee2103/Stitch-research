"""Minimal flat YAML loader (key: value lines) with no third-party dependency."""
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _parse_value(raw):
    text = raw.strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            continue
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    return text


def load_config(path=None):
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    values = {}
    for lineno, line in enumerate(config_path.read_text().splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"{config_path}:{lineno}: expected 'key: value'")
        key, raw = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{config_path}:{lineno}: empty key")
        if key in values:
            raise ValueError(f"{config_path}:{lineno}: duplicate key '{key}'")
        values[key] = _parse_value(raw)
    return values
