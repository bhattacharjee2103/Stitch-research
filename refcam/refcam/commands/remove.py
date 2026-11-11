"""remove NAME — delete a stored reference."""
import sys

from ..core.registry import register_command
from ..core.store import ReferenceStore, refs_root_from_config


@register_command("remove")
class RemoveCommand:
    help = "Delete a stored reference"

    def add_args(self, parser):
        parser.add_argument("name", help="reference name to remove")
        parser.add_argument(
            "--yes", action="store_true", help="delete without a confirmation prompt"
        )

    def run(self, args):
        store = ReferenceStore(refs_root_from_config(args.config_data))
        ReferenceStore.validate_name(args.name)
        if not store.exists(args.name):
            raise FileNotFoundError(f"reference '{args.name}' not found in {store.root}")
        if not args.yes:
            if not sys.stdin.isatty():
                raise RuntimeError(
                    "refusing to delete without --yes when stdin is not a terminal"
                )
            answer = input(f"Remove reference '{args.name}'? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("refcam: cancelled")
                return
        path = store.remove(args.name)
        print(f"refcam: removed '{args.name}' ({path})")
