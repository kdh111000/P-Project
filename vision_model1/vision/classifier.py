"""분류 모델 정의, 체크포인트 저장/로드, 추론."""

from pathlib import Path

import torch
from torch import nn
from torchvision import models

from . import config
from .dataset import build_transform, load_image


def build_model(arch, num_classes, pretrained=True):
    """torchvision 백본의 마지막 층만 클래스 수에 맞게 바꾼다."""
    model = models.get_model(arch, weights="DEFAULT" if pretrained else None)
    if hasattr(model, "fc"):                      # resnet 계열
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):            # efficientnet, mobilenet 계열
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    else:
        raise ValueError(f"지원하지 않는 구조: {arch}")
    return model


def save_checkpoint(path, model, arch, classes, img_size, **extra):
    """추론에 필요한 정보(구조, 클래스 순서, 입력 크기)를 가중치와 함께 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "arch": arch,
        "classes": list(classes),
        "img_size": img_size,
        **extra,
    }, path)


class Classifier:
    def __init__(self, ckpt_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.classes = ckpt["classes"]
        self.model = build_model(ckpt["arch"], len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.transform = build_transform(ckpt.get("img_size", config.IMG_SIZE), train=False)

    @torch.no_grad()
    def predict(self, image):
        """(예측 라벨, {라벨: 확률})을 돌려준다."""
        x = self.transform(load_image(image)).unsqueeze(0).to(self.device)
        probs = self.model(x).softmax(dim=1)[0].cpu().tolist()
        scores = dict(zip(self.classes, probs))
        return max(scores, key=scores.get), scores
