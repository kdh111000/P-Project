"""Collect and test visible empty-hanger bodies independently of hook detection."""
import argparse
from pathlib import Path

import hanger_detection as common


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "hanger_body_dataset"
RUNS = ROOT / "hanger_body_runs"


def train(args):
    for split in ("train", "val"):
        folder = args.dataset / "images" / split
        if not folder.exists() or not any(folder.glob("*.jpg")):
            raise RuntimeError(f"{split} 사진이 없습니다. collect --split {split} 명령으로 먼저 수집하세요.")
        label_dir = args.dataset / "labels" / split
        if not label_dir.exists() or not any(p.read_text(encoding="utf-8").strip() for p in label_dir.glob("*.txt")):
            raise RuntimeError(f"{split}에 몸체 박스가 표시된 사진이 없습니다.")
    args.dataset.mkdir(parents=True, exist_ok=True)
    path = str(args.dataset.resolve()).replace("\\", "/").replace("'", "''")
    config = args.dataset / "dataset.yaml"
    config.write_text(f"path: '{path}'\ntrain: images/train\nval: images/val\nnames:\n  0: empty_hanger_body\n", encoding="utf-8")
    model = common.require_yolo()(args.weights)
    model.train(data=str(config), epochs=args.epochs, imgsz=args.imgsz,
                project=str(RUNS), name="detect", exist_ok=False)
    print(f"학습 결과는 {RUNS} 폴더에 저장됩니다. best.pt 경로를 확인하세요.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("collect", help="capture labelled visible empty-hanger bodies")
    capture.add_argument("--camera", type=int, default=0)
    capture.add_argument("--split", choices=("train", "val"), required=True)
    capture.add_argument("--dataset", type=Path, default=DATASET)
    capture.set_defaults(target="empty_body")
    training = sub.add_parser("train", help="train a separate body detector")
    training.add_argument("--dataset", type=Path, default=DATASET)
    training.add_argument("--weights", default="yolov8n.pt")
    training.add_argument("--epochs", type=int, default=50)
    training.add_argument("--imgsz", type=int, default=640)
    detection = sub.add_parser("detect", help="test body detection with a camera")
    detection.add_argument("--weights", type=Path, required=True)
    detection.add_argument("--camera", type=int, default=0)
    detection.add_argument("--conf", type=float, default=0.25)
    detection.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    common.WINDOW = "Empty hanger body collection"
    try:
        {"collect": common.collect, "train": train, "detect": common.predict}[args.command](args)
    except (RuntimeError, OSError) as exc:
        parser.exit(1, f"오류: {exc}\n")


if __name__ == "__main__":
    main()
