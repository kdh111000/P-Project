"""DeepFashion2에서 종류 분류 학습용 이미지를 만든다.

    python -m vision.deepfashion2 --src <압축을 푼 폴더> --zip

<src>/image/*.jpg 와 <src>/annos/*.json 을 읽어 옷을 하나씩 잘라 data/public/<종류>/ 에 저장한다.
라벨의 윤곽선으로 옷이 아닌 픽셀을 지우므로, 등록 과정의 크롭 단계가 내보내는 이미지와 같은 형태가 된다.

DeepFashion2는 비상업적 연구 목적으로만 쓸 수 있고 재배포가 금지되어 있다.
원본과 여기서 만든 이미지 모두 저장소에 올리지 않는다.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import config
from .crop import crop_garment
from .dataset import load_image
from .detect import Detection

# DeepFashion2의 13개 종류 중 상의/하의/외투로 나눌 수 있는 것만 쓴다.
# vest(5), sling(6), 원피스류(10~13)는 세 종류 어디에도 맞지 않아 제외했다.
CATEGORY_MAP = {
    1: "top",       # short sleeve top
    2: "top",       # long sleeve top
    3: "outer",     # short sleeve outwear
    4: "outer",     # long sleeve outwear
    7: "bottom",    # shorts
    8: "bottom",    # trousers
    9: "bottom",    # skirt
}

HEAVY_OCCLUSION = 3   # occlusion: 1 약함, 2 보통, 3 심함
LARGE_ZOOM_IN = 3     # zoom_in: 1 없음, 2 보통, 3 크게 확대
MIN_SIDE = 64         # 박스의 짧은 변(픽셀). 이보다 작으면 종류를 알아보기 어렵다


@dataclass
class Item:
    image: Path
    pair_id: int
    index: int        # 한 사진 안에서의 옷 번호 (item1, item2, ...)
    category: str
    box: tuple[int, int, int, int]
    polygons: list


def find_root(src):
    """image/ 와 annos/ 가 들어 있는 폴더를 찾는다. 압축을 풀면 폴더가 한 겹 더 생기는 경우가 있다."""
    src = Path(src)
    for folder in [src, *sorted(p for p in src.glob("*") if p.is_dir()),
                   *sorted(p for p in src.glob("*/*") if p.is_dir())]:
        if (folder / "annos").is_dir() and (folder / "image").is_dir():
            return folder
    raise SystemExit(f"image/ 와 annos/ 폴더를 찾지 못했습니다: {src}")


def usable(item):
    """많이 가려졌거나 일부만 확대된 옷, 너무 작은 옷은 등록 사진과 거리가 멀어 쓰지 않는다."""
    if item.get("occlusion") == HEAVY_OCCLUSION or item.get("zoom_in") == LARGE_ZOOM_IN:
        return False
    x1, y1, x2, y2 = item["bounding_box"]
    return min(x2 - x1, y2 - y1) >= MIN_SIDE


def read_items(root):
    """라벨 파일을 모두 읽어 쓸 수 있는 옷 목록을 만든다."""
    items = []
    files = sorted((root / "annos").glob("*.json"))
    for n, path in enumerate(files, 1):
        with open(path, encoding="utf-8") as f:
            anno = json.load(f)
        for key, item in anno.items():
            if not key.startswith("item"):
                continue
            category = CATEGORY_MAP.get(item["category_id"])
            if category is None or not usable(item):
                continue
            items.append(Item(
                image=root / "image" / f"{path.stem}.jpg",
                pair_id=anno["pair_id"],
                index=int(key[4:]),
                category=category,
                box=tuple(int(v) for v in item["bounding_box"]),
                polygons=item["segmentation"],
            ))
        if n % 5000 == 0:
            print(f"  라벨 {n}/{len(files)}")
    return items


def pick_balanced(items, per_class, seed=0):
    """종류마다 최대 per_class 개를 무작위로 고른다. 한 종류가 학습을 지배하지 않게 하기 위함."""
    by_category = defaultdict(list)
    for item in items:
        by_category[item.category].append(item)

    rng = random.Random(seed)
    picked = []
    for category in config.CATEGORIES:
        pool = by_category[category]
        rng.shuffle(pool)
        picked.extend(pool[:per_class])
    return picked


def polygon_mask(polygons, size):
    """윤곽선 좌표([x1, y1, x2, y2, ...] 목록)를 이미지 크기의 bool 마스크로 바꾼다."""
    canvas = Image.new("1", size, 0)
    draw = ImageDraw.Draw(canvas)
    for poly in polygons:
        if len(poly) >= 6:   # 점이 세 개는 있어야 면이 된다
            draw.polygon(poly, fill=1)
    return np.asarray(canvas, dtype=bool)


def crop_item(item):
    image = load_image(item.image)
    mask = polygon_mask(item.polygons, image.size)
    return crop_garment(image, Detection(item.box, 1.0, item.category, mask))


def export(src, out=config.PUBLIC_DIR, per_class=3300, seed=0, make_zip=False):
    root = find_root(src)
    out = Path(out)
    for category in config.CATEGORIES:
        folder = out / category
        if folder.is_dir() and any(folder.iterdir()):
            raise SystemExit(f"이미 만들어진 이미지가 있습니다: {folder}\n"
                             "다시 만들려면 이 폴더를 지운 뒤 실행하세요.")

    print(f"라벨 읽는 중: {root / 'annos'}")
    items = read_items(root)
    available = Counter(item.category for item in items)
    picked = pick_balanced(items, per_class, seed)

    for category in config.CATEGORIES:
        (out / category).mkdir(parents=True, exist_ok=True)
    for n, item in enumerate(picked, 1):
        # 파일 이름을 pair_id로 시작해 둔다. 같은 pair_id는 같은 옷을 찍은 사진들이라,
        # 학습/검증을 나눌 때 이 값으로 묶어야 같은 옷이 양쪽에 들어가지 않는다.
        name = f"{item.pair_id:05d}_{item.image.stem}_{item.index}.jpg"
        crop_item(item).save(out / item.category / name, quality=92)
        if n % 1000 == 0:
            print(f"  저장 {n}/{len(picked)}")

    saved = Counter(item.category for item in picked)
    for category in config.CATEGORIES:
        print(f"{config.KOREAN[category]}: {saved[category]}장 저장 (쓸 수 있는 옷 {available[category]}개)")
    print(f"저장 위치: {out}")

    if make_zip:
        archive = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"압축 파일: {archive}")


def main():
    p = argparse.ArgumentParser(description="DeepFashion2에서 종류 분류 학습용 이미지 만들기")
    p.add_argument("--src", required=True, help="DeepFashion2 압축을 푼 폴더 (image/, annos/ 포함)")
    p.add_argument("--out", default=str(config.PUBLIC_DIR))
    p.add_argument("--per-class", type=int, default=3300, help="종류별 최대 장수")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zip", action="store_true", help="Colab에 올릴 압축 파일도 함께 만든다")
    args = p.parse_args()
    export(args.src, args.out, args.per_class, args.seed, args.zip)


if __name__ == "__main__":
    main()
