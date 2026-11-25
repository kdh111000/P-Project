"""등록 사진을 탐지 → 크롭 → 분류 순서로 처리한다.

    # 촬영 원본을 학습용 크롭으로 변환
    python -m vision.pipeline crop --src data/raw --dst data/crops

    # 사진 처리 결과 확인
    python -m vision.pipeline run front.jpg back.jpg --save out/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from . import config
from .crop import crop_garment
from .dataset import load_image, scan
from .detect import GarmentDetector, check_ready


@dataclass
class Result:
    image: Image.Image                 # LLM에 넘길 이미지. 크롭을 끄면 원본 그대로다
    ready: bool                        # False면 다시 찍어야 하는 사진
    reason: str
    category: str | None = None
    view: str | None = None
    category_scores: dict | None = None
    view_scores: dict | None = None
    box: tuple | None = None
    conf: float | None = None


class Pipeline:
    """모델은 처음 필요해질 때 불러온다. 가중치 파일이 없는 분류 단계는 건너뛴다."""

    def __init__(self, weight_dir=config.WEIGHT_DIR, detector=None, device=None):
        self.weight_dir = Path(weight_dir)
        self.device = device
        self._detector = detector
        self._classifiers = {}
        self._missing = set()

    @property
    def detector(self):
        if self._detector is None:
            self._detector = GarmentDetector(device=self.device)
        return self._detector

    def has(self, task):
        """분류 가중치 파일이 있는지 확인한다. 없으면 건너뛴다는 사실을 한 번만 알린다."""
        path = self.weight_dir / f"{task}.pt"
        if path.exists():
            return True
        if task not in self._missing:
            self._missing.add(task)
            print(f"[pipeline] {path} 없음 → {task} 분류를 건너뜁니다")
        return False

    def classifier(self, task):
        if task not in self._classifiers:
            from .classifier import Classifier

            path = self.weight_dir / f"{task}.pt"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} 가 없습니다. 먼저 python -m vision.train --task {task} 로 학습하세요.")
            self._classifiers[task] = Classifier(path, self.device)
        return self._classifiers[task]

    def run(self, image, use_crop=True, remove_bg=True, use_category=True, use_view=True):
        """사진 한 장을 처리한다.

        각 단계는 인자로 끌 수 있다. 단계별 기여도를 비교하는 실험에서는 같은 사진을
        설정만 바꿔 다시 돌리면 된다.
        """
        img = load_image(image)
        result = Result(image=img, ready=True, reason="ok")

        if use_crop:
            det = self.detector.detect_one(img)
            result.ready, result.reason = check_ready(det, img.size)
            if det is None:
                return result
            result.box, result.conf = det.box, det.conf
            result.image = crop_garment(img, det, remove_bg=remove_bg)

        if use_category and self.has("category"):
            result.category, result.category_scores = self.classifier("category").predict(result.image)
        if use_view and self.has("view"):
            result.view, result.view_scores = self.classifier("view").predict(result.image)
        return result

    def run_garment(self, images, **switches):
        """옷 한 벌을 찍은 여러 장을 차례로 처리한다."""
        return [self.run(img, **switches) for img in images]


def merge_category(results):
    """여러 장의 종류 확률을 평균해 옷 한 벌의 종류를 정한다.

    방향에 따라 예측이 갈릴 수 있어서 한 장의 결과만 믿지 않는다.
    """
    scored = [r.category_scores for r in results if r.category_scores]
    if not scored:
        return None
    mean = {c: sum(s[c] for s in scored) / len(scored) for c in scored[0]}
    return max(mean, key=mean.get)


def crop_dataset(src, dst, remove_bg=True):
    """촬영 원본 폴더를 같은 구조의 크롭 폴더로 변환한다. 옷을 못 찾은 사진은 건너뛴다."""
    src, dst = Path(src), Path(dst)
    detector = GarmentDetector()
    saved, missed, not_ready = 0, [], 0

    for s in scan(src):
        img = load_image(s.path)
        det = detector.detect_one(img)
        if det is None:
            missed.append(str(s.path.relative_to(src)))
            continue
        if not check_ready(det, img.size)[0]:
            not_ready += 1
        out = (dst / s.path.relative_to(src)).with_suffix(".jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        crop_garment(img, det, remove_bg=remove_bg).save(out, quality=95)
        saved += 1

    print(f"저장 {saved}장, 미탐지 {len(missed)}장, 촬영 기준 미달(저장은 함) {not_ready}장")
    if missed:
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "missed.txt").write_text("\n".join(missed), encoding="utf-8")
        print(f"미탐지 목록: {dst / 'missed.txt'}")


def main():
    p = argparse.ArgumentParser(description="의류 등록 전처리")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("crop", help="촬영 원본을 학습용 크롭으로 변환")
    c.add_argument("--src", default=str(config.RAW_DIR))
    c.add_argument("--dst", default=str(config.CROP_DIR))
    c.add_argument("--keep-bg", action="store_true", help="배경을 지우지 않고 박스만 자른다")

    r = sub.add_parser("run", help="사진을 처리하고 결과를 출력")
    r.add_argument("images", nargs="+")
    r.add_argument("--weights", default=str(config.WEIGHT_DIR))
    r.add_argument("--save", default=None, help="처리된 이미지를 저장할 폴더")
    r.add_argument("--no-crop", action="store_true")
    r.add_argument("--keep-bg", action="store_true")
    r.add_argument("--no-category", action="store_true")
    r.add_argument("--no-view", action="store_true")

    args = p.parse_args()
    if args.cmd == "crop":
        crop_dataset(args.src, args.dst, remove_bg=not args.keep_bg)
        return

    pipe = Pipeline(args.weights)
    results = pipe.run_garment(
        args.images,
        use_crop=not args.no_crop, remove_bg=not args.keep_bg,
        use_category=not args.no_category, use_view=not args.no_view,
    )
    for i, (path, res) in enumerate(zip(args.images, results), start=1):
        print(f"{path}: ready={res.ready} ({res.reason}) category={res.category} view={res.view}")
        if args.save:
            out = Path(args.save)
            out.mkdir(parents=True, exist_ok=True)
            # 방향별 폴더에서 같은 파일명이 들어올 수 있어 순번을 붙인다
            res.image.save(out / f"{i:02d}_{Path(path).stem}.jpg", quality=95)
    print(f"종류(평균): {merge_category(results)}")


if __name__ == "__main__":
    main()
