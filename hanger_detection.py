"""Collect labelled hanger photos, train YOLO, and test live detection.

Collection keys: SPACE freeze, drag boxes, Z undo, S save, N save negative,
SPACE retake, Q quit. Use separate capture sessions for train and val.
"""
import argparse
from pathlib import Path
import time

import cv2


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "hanger_dataset"
WINDOW = "Hanger dataset collection"
state = {"image": None, "boxes": [], "start": None, "cursor": None}


def on_mouse(event, x, y, flags, userdata):
    if state["image"] is None:
        return
    height, width = state["image"].shape[:2]
    x, y = max(0, min(x, width - 1)), max(0, min(y, height - 1))
    if event == cv2.EVENT_LBUTTONDOWN:
        state["start"] = (x, y)
        state["cursor"] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and state["start"] is not None:
        state["cursor"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and state["start"] is not None:
        x1, y1 = state["start"]
        x1, x2 = sorted((x1, x))
        y1, y2 = sorted((y1, y))
        if x2 - x1 >= 5 and y2 - y1 >= 5:
            state["boxes"].append((x1, y1, x2, y2))
        state["start"] = state["cursor"] = None


def write_sample(root, split, image, boxes):
    height, width = image.shape[:2]
    image_dir, label_dir = root / "images" / split, root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    name = str(time.time_ns())
    image_path = image_dir / (name + ".jpg")
    label_path = label_dir / (name + ".txt")
    rows = []
    for x1, y1, x2, y2 in boxes:
        rows.append(f"0 {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise OSError("JPEG encoding failed")
    encoded.tofile(str(image_path))
    try:
        label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    except OSError:
        image_path.unlink(missing_ok=True)
        raise
    return image_path


def write_dataset_config(root):
    root.mkdir(parents=True, exist_ok=True)
    quoted_path = str(root.resolve()).replace("\\", "/").replace("'", "''")
    path = root / "dataset.yaml"
    path.write_text(f"path: '{quoted_path}'\ntrain: images/train\nval: images/val\nnames:\n  0: hanger\n", encoding="utf-8")
    return path


def collect(args):
    target = getattr(args, "target", "hanger")
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다. --camera 1을 시도하세요.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(f"{args.split} 수집: SPACE 정지/재촬영, 마우스 드래그 라벨, Z 되돌리기, S 저장, N 빈 장면 저장, Q 종료")
    print(f"S는 보이는 모든 {target} 대상에 박스를 그린 후 저장하세요. N은 해당 대상이 하나도 없을 때만 사용하세요.")
    print("train과 val은 서로 다른 촬영 세션에서 수집하세요.")
    try:
        while True:
            if state["image"] is None:
                ok, frame = camera.read()
                if not ok:
                    print("카메라 영상을 읽지 못했습니다.")
                    break
            else:
                frame = state["image"]
            display = frame.copy()
            for x1, y1, x2, y2 in state["boxes"]:
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if state["start"] and state["cursor"]:
                cv2.rectangle(display, state["start"], state["cursor"], (0, 255, 255), 2)
            if state["image"] is None:
                lines = [f"{args.split.upper()} | LIVE: SPACE freeze | Q quit"]
            else:
                lines = [f"{args.split.upper()} | FROZEN: drag ALL {target} | boxes: {len(state['boxes'])}",
                         "S save | N save NO target | Z undo | SPACE retake | Q quit"]
            cv2.rectangle(display, (0, 0), (display.shape[1]-1, 18 + 23*len(lines)), (0, 0, 0), -1)
            for index, text in enumerate(lines):
                cv2.putText(display, text, (8, 22 + 23*index), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xff
            if key == ord('q') or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(' '):
                state["image"] = frame.copy() if state["image"] is None else None
                state["boxes"].clear()
                state["start"] = state["cursor"] = None
            elif key == ord('z') and state["image"] is not None and state["boxes"]:
                state["boxes"].pop()
            elif key in (ord('s'), ord('n')) and state["image"] is not None:
                if key == ord('s') and not state["boxes"]:
                    print(f"{target} 박스를 먼저 그리세요. 대상이 전혀 없으면 N을 누르세요.")
                    continue
                if key == ord('n') and state["boxes"]:
                    print("박스가 있습니다. 라벨이 있는 사진은 S로 저장하세요.")
                    continue
                try:
                    saved = write_sample(args.dataset, args.split, state["image"], state["boxes"])
                    print(f"저장: {saved}")
                    state["image"] = None
                    state["boxes"].clear()
                except OSError as exc:
                    print(f"저장 실패: {exc}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


def require_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError as exc:
        raise RuntimeError("YOLO가 설치되지 않았습니다. 이 환경에 ultralytics와 PyTorch를 설치해야 합니다.") from exc


def train(args):
    for split in ("train", "val"):
        images = args.dataset / "images" / split
        if not images.exists() or not any(images.glob("*.jpg")):
            raise RuntimeError(f"{split} 사진이 없습니다. collect --split {split} 명령으로 먼저 수집하세요.")
    config = write_dataset_config(args.dataset)
    YOLO = require_yolo()
    model = YOLO(args.weights)
    model.train(data=str(config), epochs=args.epochs, imgsz=args.imgsz,
                project=str(ROOT / "hanger_runs"), name="detect", exist_ok=False)
    print("학습 결과는 hanger_runs 폴더에 저장됩니다. best.pt 경로를 확인하세요.")


def predict(args):
    if not args.weights.is_file():
        raise RuntimeError(f"학습된 가중치 파일이 없습니다: {args.weights}")
    YOLO = require_yolo()
    model = YOLO(str(args.weights))
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다. --camera 1을 시도하세요.")
    cv2.namedWindow("Hanger detection", cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("카메라 영상을 읽지 못했습니다.")
                break
            result = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            display = result.plot()
            cv2.imshow("Hanger detection", display)
            if cv2.waitKey(1) & 0xff == ord('q') or cv2.getWindowProperty("Hanger detection", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("collect", help="capture and label still images")
    capture.add_argument("--camera", type=int, default=0)
    capture.add_argument("--split", choices=("train", "val"), required=True)
    capture.add_argument("--dataset", type=Path, default=DATASET)
    training = sub.add_parser("train", help="train a single-class hanger detector")
    training.add_argument("--dataset", type=Path, default=DATASET)
    training.add_argument("--weights", default="yolov8n.pt", help="pretrained weights (may require download)")
    training.add_argument("--epochs", type=int, default=50)
    training.add_argument("--imgsz", type=int, default=640)
    detection = sub.add_parser("detect", help="try trained detector on a camera")
    detection.add_argument("--weights", type=Path, required=True, help="path to trained best.pt")
    detection.add_argument("--camera", type=int, default=0)
    detection.add_argument("--conf", type=float, default=0.25)
    detection.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    try:
        {"collect": collect, "train": train, "detect": predict}[args.command](args)
    except (RuntimeError, OSError) as exc:
        parser.exit(1, f"오류: {exc}\n")


if __name__ == "__main__":
    main()
