# Adding a detector, overlay or command

refcam discovers extensions automatically. Create one file in the matching
folder, decorate the object, and it is available. **Never edit a core file to
add a feature.**

Discovery imports every module under `refcam/detectors/`, `refcam/overlays/`
and `refcam/commands/` and runs its decorators. Duplicate names raise an error
at startup, so names must be unique within their folder.

## Detector

Create `refcam/detectors/<name>.py`:

```python
from ..core.detection import Detection
from ..core.registry import register_detector

@register_detector("my_detector")
class MyDetector:
    name = "my_detector"

    def fit(self, image, bbox):
        """image: BGR ndarray; bbox: (x, y, w, h) in image pixels.
        Return any model object (kept in memory / saved to features.npz)."""
        ...

    def match(self, frame, model):
        """Return Detection(quad, confidence, inliers) or None."""
        ...
```

Contract:

- `fit(image, bbox) -> model` — analyse only the selected region.
- `match(frame, model) -> Detection | None` — `Detection.quad` is a 4x2
  float32 array ordered top-left, top-right, bottom-right, bottom-left in
  frame pixels. Return `None` when the reference is not verified.
- Select it with `detector: <name>` in `refcam/config.yaml`.

## Overlay

Create `refcam/overlays/<name>.py`:

```python
from ..core.registry import register_overlay

@register_overlay("my_style")
class MyStyle:
    def draw(self, frame, detection, ref):
        """frame: BGR ndarray; detection: verified Detection; ref: reference record.
        Return the frame with the overlay drawn (may draw in place)."""
        return frame
```

Set `overlay_style: <name>` in `refcam/config.yaml`, or cycle styles at runtime
with `o` in `run` mode.

## Command

Create `refcam/commands/<name>.py`:

```python
from ..core.registry import register_command

@register_command("hello")
class HelloCommand:
    help = "One-line help shown in --help"

    def add_args(self, parser):
        parser.add_argument("who", nargs="?", default="world")

    def run(self, args):
        print(f"hello {args.who}")
```

The subcommand is added to `python -m refcam` automatically.
`args.config_data` holds the parsed `config.yaml`. Raise `RuntimeError` with a
clear message for user-facing failures; the entry point prints it and exits 1.

## Rules

- One file per extension; no edits to `core/` or `__main__.py` required.
- Names are unique per folder.
- Keep runtime dependencies to NumPy and OpenCV unless agreed otherwise.
- Failures must be explicit: never return a silent empty result when the
  reference is missing or unreadable.

## Current status

| Extension point | Folder | Available |
|---|---|---|
| Commands | `refcam/commands/` | every command file is auto-registered |
| Detectors | `refcam/detectors/` | `sift.fit()` and `sift.match()` use SIFT descriptors and a RANSAC homography |
| Overlays | `refcam/overlays/` | `outline` draws the detected quad; `project` maps an attached image onto it |

Reference storage is implemented in `core/store.py`: each reference
is `refs/<name>/` with `crop.png`, `meta.json`, `features.npz` and an optional
`overlay.png`.
