"""overlay NAME PATH_TO_PNG — attach or replace a reference overlay."""
from ..core.registry import register_command
from ..core.store import ReferenceStore, refs_root_from_config


@register_command("overlay")
class OverlayCommand:
    help = "Attach or replace the overlay picture for a reference"

    def add_args(self, parser):
        parser.add_argument("name", help="reference name")
        parser.add_argument("image", help="path to a PNG/JPG image (alpha is preserved)")

    def run(self, args):
        store = ReferenceStore(refs_root_from_config(args.config_data))
        target = store.set_overlay(args.name, args.image)
        print(f"refcam: overlay attached to '{args.name}' -> {target}")
