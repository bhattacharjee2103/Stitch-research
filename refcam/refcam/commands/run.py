"""Live preview with geometric reference detection and perspective overlays."""
import time
from collections import deque

import cv2
import numpy as np

from ..core.camera import FrameSource
from ..core.detection import Detection
from ..core.display import draw_label, fit_frame
from ..core.registry import (
    DETECTORS,
    OVERLAYS,
    load_detectors,
    load_overlays,
    register_command,
)
from ..core.store import ReferenceStore, refs_root_from_config


def _load_references(store, match_ratio, min_inliers):
    """Load every usable saved reference and its crop dimensions."""
    references = []
    for summary in store.summaries():
        if summary.error:
            print(f"refcam: skipping reference '{summary.name}': {summary.error}")
            continue
        try:
            reference = store.load(summary.name)
            crop = cv2.imread(str(reference.crop_path), cv2.IMREAD_COLOR)
            if crop is None:
                raise ValueError("crop.png is missing or unreadable")
            if len(reference.keypoints) < 4 or len(reference.descriptors) < 4:
                raise ValueError("not enough saved SIFT features")
            model = {
                "keypoints": reference.keypoints,
                "descriptors": reference.descriptors,
                "size": (crop.shape[1], crop.shape[0]),
                "match_ratio": match_ratio,
                "min_inliers": min_inliers,
            }
            references.append((reference, model))
        except Exception as exc:
            print(f"refcam: skipping reference '{summary.name}': {exc}")
    return references


def _make_plugin(entry):
    return entry() if isinstance(entry, type) else entry


def _capture_reference(
    frame, detector, store, source_label, min_keypoints, match_ratio, min_inliers
):
    """Select and save a reference from the current live-preview frame."""
    while True:
        window = "refcam | capture - drag box, ENTER accepts, c cancels"
        print("refcam: drag a box around the object and press ENTER (c cancels)", flush=True)
        try:
            x, y, box_width, box_height = cv2.selectROI(
                window, frame, showCrosshair=True, fromCenter=False
            )
        finally:
            cv2.destroyWindow(window)
        if box_width <= 0 or box_height <= 0:
            print("refcam: capture cancelled")
            return None

        bbox = (int(x), int(y), int(box_width), int(box_height))
        crop = frame[y : y + box_height, x : x + box_width].copy()
        try:
            model = detector.fit(frame, bbox)
        except (ValueError, cv2.error) as exc:
            print(f"refcam: could not analyse that selection: {exc}")
            continue
        keypoint_count = len(model.get("keypoints", []))
        if keypoint_count < min_keypoints:
            print(
                f"refcam: that selection has {keypoint_count} features; "
                f"choose a more detailed region (need {min_keypoints})"
            )
            continue
        break

    while True:
        try:
            name = input("Reference name: ").strip()
        except EOFError:
            print("refcam: could not read a name; run this command from Terminal")
            return None
        try:
            ReferenceStore.validate_name(name)
        except ValueError as exc:
            print(f"refcam: {exc}")
            continue
        if store.exists(name):
            print(f"refcam: reference '{name}' already exists; choose another name")
            continue
        break

    saved = store.save(
        name,
        crop,
        model,
        bbox,
        (frame.shape[1], frame.shape[0]),
        source_label,
    )
    runtime_model = dict(model)
    runtime_model.update(
        {
            "size": (crop.shape[1], crop.shape[0]),
            "match_ratio": match_ratio,
            "min_inliers": min_inliers,
        }
    )
    print(f"refcam: saved '{name}' with {keypoint_count} features", flush=True)
    return saved, runtime_model


def _attach_overlay(references, store, active):
    """Prompt for a saved reference and attach an image to it."""
    if not references:
        print("refcam: capture a reference first by pressing C")
        return False
    names = [reference.name for reference, _ in references]
    default_name = active["reference"].name if active is not None else names[0]
    try:
        name = input(f"Reference name [{default_name}]: ").strip() or default_name
        image_path = input("Overlay image path: ").strip()
    except EOFError:
        print("refcam: could not read input; run this command from Terminal")
        return False
    if name not in names:
        print(f"refcam: unknown reference '{name}' (available: {', '.join(names)})")
        return False
    if not image_path:
        print("refcam: no overlay image selected")
        return False
    try:
        target = store.set_overlay(name, image_path)
    except (OSError, ValueError) as exc:
        print(f"refcam: could not attach overlay: {exc}")
        return False
    print(f"refcam: overlay attached to '{name}' -> {target}", flush=True)
    return True


@register_command("run")
class RunCommand:
    help = "Live detection; press C to capture a reference or P to attach an image"

    def add_args(self, parser):
        parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
        parser.add_argument("--video", default=None, help="video file instead of a camera")
        parser.add_argument("--image", default=None, help="still image instead of a camera")

    def run(self, args):
        config = args.config_data
        width = int(config.get("resolution_width", 640))
        height = int(config.get("resolution_height", 480))
        window_size = int(config.get("fps_window", 30))
        match_ratio = min(0.99, max(0.1, float(config.get("match_ratio", 0.75))))
        min_inliers = max(4, int(config.get("min_inliers", 12)))
        min_keypoints = max(4, int(config.get("min_keypoints", 40)))
        hold_frames = max(0, int(config.get("hold_frames", 15)))
        ema_alpha = min(1.0, max(0.0, float(config.get("ema_alpha", 0.6))))

        load_detectors()
        detector_entry = DETECTORS.get(str(config.get("detector", "sift")))
        detector = _make_plugin(detector_entry)
        store = ReferenceStore(refs_root_from_config(config))
        references = _load_references(store, match_ratio, min_inliers)

        load_overlays()
        style_names = OVERLAYS.names()
        if not style_names:
            raise RuntimeError("no overlay styles are registered")
        style_name = str(config.get("overlay_style", "outline"))
        if style_name not in style_names:
            raise ValueError(
                f"unknown overlay style '{style_name}' (available: {', '.join(style_names)})"
            )
        style_index = style_names.index(style_name)
        styles = {name: _make_plugin(OVERLAYS.get(name)) for name in style_names}

        source = FrameSource(args.camera, args.video, args.image).open()
        window = "refcam | live preview"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, width, height)
        print(f"refcam: {source.source_label} — press q or Esc to quit", flush=True)
        if references:
            print(
                "refcam: watching for "
                + ", ".join(reference.name for reference, _ in references),
                flush=True,
            )
        else:
            print("refcam: press C in the preview to capture a reference", flush=True)

        timestamps = deque(maxlen=max(2, window_size))
        tracked = None
        misses = 0
        read_misses = 0
        try:
            while True:
                ok, frame = source.read()
                if not ok:
                    if source.is_video:
                        print("refcam: end of video")
                        break
                    read_misses += 1
                    if read_misses >= 60:
                        raise RuntimeError(
                            f"no frames received from {source.source_label}. "
                            "Check macOS camera permission, then try another --camera INDEX."
                        )
                    time.sleep(0.03)
                    continue
                read_misses = 0

                camera_frame = fit_frame(frame, width, height)
                preview = camera_frame.copy()
                best = None
                for reference, model in references:
                    detection = detector.match(camera_frame, model)
                    if detection is None:
                        continue
                    score = detection.confidence * min(detection.inliers, 50)
                    if best is None or score > best[0]:
                        best = (score, reference, detection)

                if best is not None:
                    _, reference, detection = best
                    if tracked is not None and tracked["reference"].name == reference.name:
                        quad = (
                            (1.0 - ema_alpha) * tracked["detection"].quad
                            + ema_alpha * detection.quad
                        ).astype(np.float32)
                        detection = Detection(quad, detection.confidence, detection.inliers)
                    tracked = {"reference": reference, "detection": detection}
                    misses = 0
                elif tracked is not None:
                    misses += 1
                    if misses > hold_frames:
                        tracked = None

                active = tracked
                timestamps.append(time.perf_counter())
                if len(timestamps) > 1:
                    elapsed = timestamps[-1] - timestamps[0]
                    fps = (len(timestamps) - 1) / elapsed if elapsed > 0 else 0.0
                else:
                    fps = 0.0

                if active is not None:
                    styles[style_name].draw(
                        preview, active["detection"], active["reference"]
                    )
                draw_label(preview, f"FPS {fps:5.1f}  |  {source.source_label}", 28)
                if active is not None:
                    suffix = "  (held)" if misses else ""
                    draw_label(
                        preview,
                        f"Found {active['reference'].name}{suffix}",
                        54,
                        (150, 255, 160),
                    )
                elif references:
                    draw_label(preview, "Looking for saved references...", 54, (210, 220, 255))
                else:
                    draw_label(preview, "Add a reference to enable detection", 54, (210, 220, 255))
                draw_label(preview, f"C capture  |  P attach image  |  O style  |  Q quit", 80)

                cv2.imshow(window, preview)
                key = cv2.waitKey(1 if not source.is_image else 30) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("o"), ord("O")):
                    style_index = (style_index + 1) % len(style_names)
                    style_name = style_names[style_index]
                    print(f"refcam: overlay style is now '{style_name}'", flush=True)
                if key in (ord("c"), ord("C")):
                    result = _capture_reference(
                        camera_frame.copy(),
                        detector,
                        store,
                        source.source_label,
                        min_keypoints,
                        match_ratio,
                        min_inliers,
                    )
                    if result is not None:
                        references.append(result)
                        tracked = None
                        misses = 0
                if key in (ord("p"), ord("P")):
                    if _attach_overlay(references, store, active) and "project" in styles:
                        style_name = "project"
                        style_index = style_names.index(style_name)
        finally:
            source.release()
            cv2.destroyAllWindows()
