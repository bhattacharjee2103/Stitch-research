"""test — run the pytest suite from the project root."""
import importlib.util
import subprocess
import sys
from pathlib import Path

from ..core.registry import register_command

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@register_command("test")
class TestCommand:
    help = "Run the pytest suite"

    def add_args(self, parser):
        parser.add_argument(
            "pytest_args",
            nargs="*",
            help="extra arguments forwarded to pytest (e.g. -k add)",
        )

    def run(self, args):
        if importlib.util.find_spec("pytest") is None:
            raise RuntimeError(
                "pytest is not installed. Install dev requirements: "
                ".venv/bin/pip install -r requirements-dev.txt"
            )
        command = [sys.executable, "-m", "pytest", "-q", *args.pytest_args]
        raise SystemExit(subprocess.call(command, cwd=PROJECT_ROOT))
