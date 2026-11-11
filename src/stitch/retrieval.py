"""CosPlace ResNet18/512 global descriptors; compatible with official weights."""
from pathlib import Path
import hashlib
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

class L2Norm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=1)

class GeM(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=1e-6).pow(self.p).mean(dim=(-2,-1),keepdim=True).pow(1/self.p)

class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:,:,0,0]

class CosPlaceNet(nn.Module):
    """Official architecture rebuilt without an unnecessary ImageNet download."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet18(weights=None).children())[:-2])
        self.aggregation = nn.Sequential(L2Norm(),GeM(),Flatten(),nn.Linear(512,512),L2Norm())

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return self.aggregation(self.backbone(x))

def image_tensor(image: np.ndarray, size: int = 224) -> torch.Tensor:
    rgb = cv2.cvtColor(cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/255
    return ((t-torch.tensor([.485,.456,.406])[:,None,None]) /
            torch.tensor([.229,.224,.225])[:,None,None])[None]

class GlobalEngine:
    def __init__(self, weights: str = 'models/cosplace_r18_512.pth', device: str = 'cpu',
                 enhance: bool = False, size: int = 224):
        if not Path(weights).is_file():
            raise FileNotFoundError('Missing pretrained weights; run scripts/download_models.py')
        self.model = CosPlaceNet()
        self.weights_sha256 = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
        self.model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True), strict=True)
        self.model = self.model.eval().to(device)
        self.device,self.enhance,self.size = device,enhance,size

    @torch.inference_mode()
    def extract(self, image: np.ndarray) -> np.ndarray:
        if self.enhance:
            lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
            lab[:,:,0]=cv2.createCLAHE(2.0,(8,8)).apply(lab[:,:,0])
            image=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
        return self.model(image_tensor(image,self.size).to(self.device))[0].cpu().numpy()

def rank(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """Stable cosine ranking; descriptors are L2 normalized."""
    return np.argsort(-(database @ query), kind='stable')
