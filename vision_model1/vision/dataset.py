"""이미지 로딩과 학습용 데이터셋.

폴더 구조에서 라벨을 읽는다.

    <root>/<옷ID>_<종류>/<방향>/*.jpg      예) data/crops/03_outer/back/0007.jpg
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

from . import config

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_image(src):
    """경로나 PIL 이미지를 RGB로 통일한다. 폰 사진은 EXIF 회전값을 반영해야 눕지 않는다."""
    img = src if isinstance(src, Image.Image) else Image.open(src)
    return ImageOps.exif_transpose(img).convert("RGB")


@dataclass
class Sample:
    path: Path
    garment: str             # 옷 ID. 같은 옷의 사진은 학습/검증 중 한쪽에만 들어간다
    category: str | None
    view: str | None


def scan(root):
    """<옷ID>_<종류>/<방향>/ 구조를 훑어 샘플 목록을 만든다. 규칙에 안 맞는 폴더는 건너뛴다."""
    root = Path(root)
    samples, ignored = [], []
    if not root.is_dir():
        return samples
    for garment_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        garment, _, category = garment_dir.name.rpartition("_")
        if not garment or category not in config.CATEGORIES:
            ignored.append(garment_dir.name)
            continue
        for view_dir in sorted(p for p in garment_dir.iterdir() if p.is_dir()):
            if view_dir.name not in config.VIEWS:
                ignored.append(f"{garment_dir.name}/{view_dir.name}")
                continue
            for f in sorted(view_dir.iterdir()):
                if f.suffix.lower() in IMG_EXT:
                    samples.append(Sample(f, garment_dir.name, category, view_dir.name))
    if ignored:
        print(f"[dataset] 이름 규칙에 맞지 않아 제외: {', '.join(ignored)}")
    return samples


def scan_flat(root):
    """<종류>/*.jpg 구조. 종류 라벨만 있는 공개 데이터를 학습에 쓸 때 사용한다.

    파일 이름에서 첫 '_' 앞부분을 읽어 옷 ID로 쓴다. vision.deepfashion2 가 같은 옷을 찍은
    사진들에 같은 앞부분을 붙여 저장하므로, 학습/검증을 나눌 때 한쪽으로 묶인다.
    """
    root = Path(root)
    samples = []
    for category in config.CATEGORIES:
        folder = root / category
        if not folder.is_dir():
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                samples.append(Sample(f, f"{f.stem.split('_')[0]}_{category}", category, None))
    return samples


def split_by_garment(samples, val_ratio=0.2, seed=0):
    """옷 단위로 학습/검증을 나눈다.

    사진 단위로 섞으면 같은 옷이 양쪽에 들어가 검증 정확도가 실제보다 높게 나온다.
    종류별로 따로 뽑아서 검증셋에 세 종류가 모두 들어가게 한다.
    """
    by_category = defaultdict(set)
    for s in samples:
        by_category[s.category].add(s.garment)

    rng = random.Random(seed)
    val_garments = set()
    for category in sorted(by_category):
        garments = sorted(by_category[category])
        rng.shuffle(garments)
        if len(garments) < 2:
            continue  # 한 벌뿐이면 검증으로 빼지 않는다
        n_val = max(1, round(len(garments) * val_ratio))
        val_garments.update(garments[:n_val])

    train = [s for s in samples if s.garment not in val_garments]
    val = [s for s in samples if s.garment in val_garments]
    return train, val


def build_transform(img_size=config.IMG_SIZE, train=False):
    if not train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    # 좌우 반전은 앞/뒤 라벨을 바꾸지 않으므로 써도 된다. 상하 반전은 쓰지 않는다.
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        T.RandomRotation(10, fill=255),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class GarmentDataset(Dataset):
    """task가 'category'면 종류, 'view'면 방향을 정답으로 돌려준다."""

    def __init__(self, samples, task, img_size=config.IMG_SIZE, train=False):
        if task not in ("category", "view"):
            raise ValueError(f"task는 category 또는 view여야 한다: {task}")
        self.task = task
        self.classes = config.CATEGORIES if task == "category" else config.VIEWS
        self.samples = [s for s in samples if getattr(s, task) is not None]
        self.transform = build_transform(img_size, train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        x = self.transform(load_image(s.path))
        y = self.classes.index(getattr(s, self.task))
        return x, torch.tensor(y)
