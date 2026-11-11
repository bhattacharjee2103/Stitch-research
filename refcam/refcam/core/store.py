"""On-disk reference storage: refs/<name>/ with crop, meta, features and overlay."""
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
SCHEMA_VERSION = 1


@dataclass
class Reference:
    name: str
    path: Path
    meta: dict
    keypoints: np.ndarray
    descriptors: np.ndarray

    @property
    def crop_path(self):
        return self.path / "crop.png"

    @property
    def overlay_path(self):
        candidate = self.path / "overlay.png"
        return candidate if candidate.is_file() else None


@dataclass
class ReferenceSummary:
    name: str
    keypoints: int | None
    has_overlay: bool
    error: str | None = None


def refs_root_from_config(config):
    """Resolve refs_dir relative to the project root, not the caller's cwd."""
    raw = str(config.get("refs_dir", "refs"))
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


class ReferenceStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_name(name):
        if not isinstance(name, str) or not NAME_PATTERN.match(name):
            raise ValueError(
                "reference name must use only letters, digits, '-' and '_' (got %r)" % (name,)
            )
        return name

    def path_for(self, name):
        self.validate_name(name)
        return self.root / name

    def exists(self, name):
        return self.path_for(name).is_dir()

    def save(self, name, crop, model, bbox, frame_size, source=""):
        self.validate_name(name)
        final = self.path_for(name)
        if final.exists():
            raise FileExistsError(
                f"reference '{name}' already exists; remove it first: "
                f"python -m refcam remove {name}"
            )
        if crop is None or getattr(crop, "size", 0) == 0:
            raise ValueError("reference crop is empty")
        keypoints = np.asarray(model.get("keypoints"), dtype=np.float32).reshape(-1, 2)
        descriptors = np.asarray(model.get("descriptors"), dtype=np.float32)
        if descriptors.ndim != 2 or descriptors.shape[0] != keypoints.shape[0]:
            raise ValueError("detector model has mismatched keypoints and descriptors")
        meta = {
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "detector": model.get("detector", "unknown"),
            "bbox": [int(v) for v in bbox],
            "frame_size": [int(v) for v in frame_size],
            "keypoints": int(keypoints.shape[0]),
            "source": source,
        }
        temporary = Path(tempfile.mkdtemp(prefix=f".tmp-{name}-", dir=self.root))
        try:
            if not cv2.imwrite(str(temporary / "crop.png"), crop):
                raise OSError("could not write crop.png")
            np.savez_compressed(
                temporary / "features.npz",
                keypoints=keypoints,
                descriptors=descriptors,
            )
            (temporary / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            os.replace(temporary, final)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return Reference(name, final, meta, keypoints, descriptors)

    def load(self, name):
        path = self.path_for(name)
        if not path.is_dir():
            raise FileNotFoundError(f"reference '{name}' not found in {self.root}")
        meta_path = path / "meta.json"
        features_path = path / "features.npz"
        if not meta_path.is_file() or not features_path.is_file():
            raise ValueError(
                f"reference '{name}' is incomplete (missing meta.json or features.npz)"
            )
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"reference '{name}' has invalid meta.json: {exc}") from exc
        with np.load(features_path) as data:
            keypoints = data["keypoints"].astype(np.float32)
            descriptors = data["descriptors"].astype(np.float32)
        return Reference(name, path, meta, keypoints, descriptors)

    def summaries(self):
        items = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or path.name.startswith("."):
                continue
            has_overlay = (path / "overlay.png").is_file()
            try:
                meta = json.loads((path / "meta.json").read_text())
                keypoints = meta.get("keypoints")
                if keypoints is None:
                    raise ValueError("meta.json has no keypoint count")
                items.append(
                    ReferenceSummary(path.name, int(keypoints), has_overlay, None)
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                items.append(ReferenceSummary(path.name, None, has_overlay, str(exc)))
        return items

    def set_overlay(self, name, image_path):
        reference = self.load(name)
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"cannot read overlay image: {image_path}")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        elif image.shape[2] != 4:
            raise ValueError("overlay must be a grayscale, RGB or RGBA image")
        if min(image.shape[:2]) < 2:
            raise ValueError("overlay image is too small")
        target = reference.path / "overlay.png"
        if not cv2.imwrite(str(target), image):
            raise OSError(f"could not write overlay for '{name}'")
        return target

    def remove(self, name):
        path = self.path_for(name)
        if not path.is_dir():
            raise FileNotFoundError(f"reference '{name}' not found in {self.root}")
        shutil.rmtree(path)
        return path
