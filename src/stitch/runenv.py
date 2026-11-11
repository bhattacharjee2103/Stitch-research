"""Run-environment selection for research scripts.

STITCH_CONFIG  config file (default configs/default.yaml, the frozen CPU setup)
STITCH_RESULTS output directory (default results, the frozen thesis evidence)

Defaults reproduce the original CPU runs exactly; set e.g.
STITCH_CONFIG=configs/gpu.yaml STITCH_RESULTS=results_gpu for a separate GPU run.
"""
import os
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[2]


def config_path() -> Path:
    return ROOT / os.environ.get('STITCH_CONFIG', 'configs/default.yaml')


def load_config() -> Dict[str, Any]:
    return yaml.safe_load(config_path().read_text())


def results_dir() -> Path:
    path = ROOT / os.environ.get('STITCH_RESULTS', 'results')
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_default_results() -> bool:
    return results_dir().resolve() == (ROOT / 'results').resolve()


def artifact_dir(default: str) -> Path:
    """Where generated models/videos go: the original location for the default
    run, otherwise a subfolder of the alternate results directory."""
    path = ROOT / default if is_default_results() else results_dir() / Path(default).name
    path.mkdir(parents=True, exist_ok=True)
    return path


def device(config: Dict[str, Any]) -> str:
    return config.get('device', 'cpu')
