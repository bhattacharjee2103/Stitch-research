"""add NAME — scan, freeze, select a region, analyse and save a reference."""
import time

import cv2

from ..core.camera import FrameSource
from ..core.display import draw_label, fit_frame
from ..core.registry import DETECTORS, load_detectors, register_command
from ..core.store import ReferenceStore, refs_root_from_config

MIN_BOX_PX = 8


@register_command("add")
class AddCommand:
    help = "Scan, select a region and save it as a reference"

    def add_args(self, parser):
        parser.add_argument(
            "name",
            nargs="?",
            default=None,
            help="reference name (prompted after selection if omitted)",
        )
        parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
        parser.add_argument("--video", default=None, help="video file instead of a camera")
        parser.add_argument("--image", default=None, help="still image instead of a camera")

    def run(self, args):
        config = args.config_data
        width = int(config.get("resolution_width", 640))
        height = int(config.get("resolution_height", 480))
        min_keypoints = int(config.get("min_keypoints", 40))
        load_detectors()
        detector_entry = DETECTORS.get(str(config.get("detector", "sift")))
        detector = detector_entry() if isinstance(detector_entry, type) else detector_entry
        store = ReferenceStore(refs_root_from_config(config))

        source = FrameSource(args.camera, args.video, args.image).open()
        try:
            frozen = self._freeze(source, width, height)
        finally:
            source.release()
        if frozen is None:
            print("refcam: no frame frozen; nothing saved")
            return
        self._add_loop(
            frozen, args.name, detector, store, min_keypoints, source.source_label
        )

    def _freeze(self, source, width, height):
        if source.is_image:
            ok, frame = source.read()
            if not ok or frame is None:
                raise RuntimeError(f"cannot read image: {source.image}")
            return fit_frame(frame, width, height)

        window = "refcam | add - SPACE freezes, q quits"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, width, height)
        print("refcam: press SPACE to freeze the frame (q or Esc quits)", flush=True)
        misses = 0
        frozen = None
        try:
            while True:
                ok, frame = source.read()
                if not ok:
                    if source.is_video:
                        raise RuntimeError("video ended before a frame was frozen")
                    misses += 1
                    if misses >= 60:
                        raise RuntimeError(
                            f"no frames received from {source.source_label}. "
                            "Check macOS camera permission, then try another --camera INDEX."
                        )
                    time.sleep(0.03)
                    continue
                misses = 0
                preview = fit_frame(frame, width, height)
                draw_label(preview, "SPACE freeze  |  q quit", 28)
                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    frozen = preview.copy()
                    break
                if key in (ord("q"), 27):
                    break
        finally:
            cv2.destroyWindow(window)
        return frozen

    def _select_box(self, frame):
        window = "refcam | draw box, ENTER accepts, c cancels"
        print("refcam: draw a box and press ENTER (c cancels)", flush=True)
        try:
            x, y, w, h = cv2.selectROI(
                window, frame, showCrosshair=True, fromCenter=False
            )
        finally:
            cv2.destroyWindow(window)
        if w <= 0 or h <= 0:
            return None
        return int(x), int(y), int(w), int(h)

    def _add_loop(self, frame, name, detector, store, min_keypoints, source_label):
        while True:
            bbox = self._select_box(frame)
            if bbox is None:
                print("refcam: selection cancelled; nothing saved")
                return
            x, y, w, h = bbox
            if w < MIN_BOX_PX or h < MIN_BOX_PX:
                print(f"refcam: selection is too small ({w}x{h}px); draw a larger box")
                continue
            if name is None:
                try:
                    name = input("Reference name: ").strip()
                except EOFError:
                    raise RuntimeError("a reference name is required")
            if not name:
                raise RuntimeError("a reference name is required")
            ReferenceStore.validate_name(name)
            if store.exists(name):
                raise FileExistsError(
                    f"reference '{name}' already exists; remove it first: "
                    f"python -m refcam remove {name}"
                )

            crop = frame[y : y + h, x : x + w].copy()
            model = detector.fit(frame, bbox)
            keypoints = int(len(model.get("keypoints", [])))
            if keypoints < min_keypoints:
                print(
                    f"refcam: low texture — {keypoints} keypoints (need {min_keypoints}). "
                    "Pick a larger or more detailed region and draw again."
                )
                continue

            saved = store.save(
                name,
                crop,
                model,
                bbox,
                (frame.shape[1], frame.shape[0]),
                source_label,
            )
            print(f"refcam: saved reference '{name}' — {keypoints} keypoints")
            print(f"  {saved.path}")
            print(
                "  optional overlay: python -m refcam overlay "
                f"{name} path/to/image.png"
            )
            print("  run detection: python -m refcam run")
            return
