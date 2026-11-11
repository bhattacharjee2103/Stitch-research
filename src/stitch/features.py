"""Classical and pretrained sparse features with a shared pixel convention."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import sys
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
torch.hub.set_dir(str(ROOT/'models'/'torch'/'hub'))
sys.path.insert(0, str(ROOT/'vendor'/'LightGlue'))

@dataclass
class Features:
    points: np.ndarray
    descriptors: np.ndarray
    raw: Dict[str, Any]

class LocalEngine:
    """Extract/match SIFT+FLANN or SuperPoint+LightGlue, on the same images."""
    def __init__(self, method: str = 'sift', max_keypoints: int = 1024,
                 ratio: float = 0.70, device: str = 'cpu', enhance: bool = False,
                 depth_confidence: float = .95, width_confidence: float = .99):
        if method not in ('sift', 'superpoint'):
            raise ValueError(method)
        self.method, self.device, self.ratio = method, device, ratio
        self.enhance = enhance
        if method == 'sift':
            self.extractor = cv2.SIFT_create(nfeatures=max_keypoints)
            self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        else:
            from lightglue import SuperPoint, LightGlue
            self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
            self.matcher = LightGlue(features='superpoint', flash=False,
                                    depth_confidence=depth_confidence,
                                    width_confidence=width_confidence).eval().to(device)

    @torch.inference_mode()
    def extract(self, image: np.ndarray) -> Features:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.enhance:
            gray = cv2.createCLAHE(2.0,(8,8)).apply(gray)
        if self.method == 'sift':
            points, descriptors = self.extractor.detectAndCompute(gray, None)
            xy = np.array([p.pt for p in points], np.float32).reshape(-1,2)
            descriptors = descriptors if descriptors is not None else np.empty((0,128),np.float32)
            return Features(xy, descriptors, {})
        tensor = torch.from_numpy(gray.copy()).float()[None]/255
        feats = self.extractor.extract(tensor.to(self.device), resize=None)
        return Features(feats['keypoints'][0].cpu().numpy(),
                        feats['descriptors'][0].cpu().numpy(), feats)

    @torch.inference_mode()
    def match(self, a: Features, b: Features) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        if len(a.points) == 0 or len(b.points) < 2:
            empty = np.empty((0,2),np.float32)
            return empty, empty.copy(), np.empty((0,2),int)
        if self.method == 'sift':
            nearest = self.matcher.knnMatch(a.descriptors, b.descriptors, k=2)
            pairs = [(p[0].queryIdx,p[0].trainIdx) for p in nearest
                     if len(p)==2 and p[0].distance < self.ratio*p[1].distance]
            indices = np.asarray(pairs, dtype=int).reshape(-1,2)
        else:
            out = self.matcher({'image0':a.raw,'image1':b.raw})
            indices = out['matches'][0].cpu().numpy().reshape(-1,2)
        return a.points[indices[:,0]], b.points[indices[:,1]], indices
