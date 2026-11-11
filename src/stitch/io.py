"""Image and video I/O with explicit coordinate transforms."""
from pathlib import Path
from typing import Iterator, Tuple, Union
import cv2
import numpy as np

def read_image(path: Union[str, Path], max_side: int = 640) -> Tuple[np.ndarray, np.ndarray]:
    """Return BGR image and original-to-resized pixel transform."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Cannot decode image: {path}')
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w)) if max_side else 1.0
    nh, nw = max(1, round(h*scale)), max(1, round(w*scale))
    if (nh, nw) != (h, w):
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return image, np.diag([nw/w, nh/h, 1.0])

def frames(source: str, limit: int = 0) -> Iterator[np.ndarray]:
    """Read file, webcam index, or stream; stop cleanly at EOF or failure."""
    cap = cv2.VideoCapture(int(source) if source.isdecimal() else source)
    if not cap.isOpened():
        raise OSError(f'Cannot open video input: {source}')
    try:
        count = 0
        while not limit or count < limit:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            yield frame
    finally:
        cap.release()
