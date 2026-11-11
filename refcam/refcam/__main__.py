"""CLI entry point. Parsing and dispatch only; commands live in commands/."""
import argparse
import sys

from .core.config import load_config
from .core.registry import load_commands


def build_parser():
    parser = argparse.ArgumentParser(
        prog="refcam",
        description="Reference-object overlay on a live camera.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="path to config.yaml (defaults to refcam/config.yaml)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, command in load_commands().items():
        instance = command() if isinstance(command, type) else command
        subparser = subparsers.add_parser(name, help=getattr(instance, "help", None))
        instance.add_args(subparser)
        subparser.set_defaults(command_obj=instance)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    command = getattr(args, "command_obj", None)
    if command is None:  # argparse normally prevents this
        print("refcam: error: no command selected", file=sys.stderr)
        return 2
    try:
        # Load once here so every command gets the same config object.
        args.config_data = load_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"refcam: error: cannot load config: {exc}", file=sys.stderr)
        return 2
    try:
        command.run(args)
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f"refcam: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
