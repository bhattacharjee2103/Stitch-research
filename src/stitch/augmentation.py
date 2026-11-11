"""Seeded photometric stressors; synthetic images are not real night captures."""
from typing import Tuple
import cv2
import numpy as np

def appearance(image: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    x=image.astype(np.float32)/255
    if condition=='day':
        x=np.clip(x*float(rng.uniform(.85,1.1)),0,1)
    elif condition in ('night','night_severe','warm_night'):
        gamma,gain,sigma=(2.6,.28,.018) if condition=='night_severe' else (1.8,.60,.010)
        x=np.power(x,gamma)*gain
        if condition=='warm_night':x=x*np.array([.65,.85,1.10],np.float32)
        x=x+rng.normal(0,sigma,x.shape).astype(np.float32)
    elif condition=='fog':
        x=.55*x+.45*np.ones_like(x)*.8
        x=cv2.GaussianBlur(x,(3,3),.7)
    else:raise ValueError(condition)
    return (np.clip(x,0,1)*255).astype(np.uint8)

def warp(image: np.ndarray, rng: np.random.Generator, strength: float = .055) -> Tuple[np.ndarray,np.ndarray]:
    h,w=image.shape[:2]
    source=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32)
    target=source+rng.uniform(-strength,strength,(4,2)).astype(np.float32)*np.array([w,h],np.float32)
    H=cv2.getPerspectiveTransform(source,target)
    return cv2.warpPerspective(image,H,(w,h)),H
