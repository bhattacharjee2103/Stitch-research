"""list — show stored references, keypoint counts and overlay status."""
from ..core.registry import register_command
from ..core.store import ReferenceStore, refs_root_from_config


@register_command("list")
class ListCommand:
    help = "Show stored references, keypoint counts and overlay status"

    def add_args(self, parser):
        pass

    def run(self, args):
        store = ReferenceStore(refs_root_from_config(args.config_data))
        summaries = store.summaries()
        if not summaries:
            print("refcam: no references yet — add one: python -m refcam add NAME")
            return
        print(f"{'NAME':<24} {'KEYPOINTS':>10} {'OVERLAY':>8}")
        print("-" * 46)
        for item in summaries:
            keypoints = "?" if item.keypoints is None else str(item.keypoints)
            overlay = "yes" if item.has_overlay else "no"
            print(f"{item.name:<24} {keypoints:>10} {overlay:>8}")
            if item.error:
                print(f"  ! unreadable: {item.error}")
        print(f"\n{len(summaries)} reference(s) in {store.root}")
