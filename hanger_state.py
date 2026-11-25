"""Use a trained hook detector to collect and classify hanger state crops."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2

from hanger_detection import require_yolo


ROOT = Path(__file__).resolve().parent
HOOK_WEIGHTS = ROOT / "hanger_runs" / "detect-2" / "weights" / "best.pt"
DATASET = ROOT / "hanger_state_dataset"
RUNS = ROOT / "hanger_state_runs"
WINDOW = "Hanger state"
STATE_IMAGE_SIZE = 224
EMPTY = "empty_hanger"
LOADED = "with_clothes"
LABELS = {EMPTY: "Empty", LOADED: "With clothes", "unknown": "Unclear"}
COLORS = {EMPTY: (100, 225, 90), LOADED: (255, 195, 70), "unknown": (0, 215, 255)}
KEY_LABELS = {ord("1"): EMPTY, ord("2"): LOADED}


@dataclass
class Region:
    hook: tuple
    bounds: tuple

    def crop(self, frame):
        x1, y1, x2, y2 = self.bounds
        image = frame[y1:y2, x1:x2]
        height, width = image.shape[:2]
        # Pad to square so classification preprocessing keeps the entire crop.
        pad_y, pad_x = max(0, width - height), max(0, height - width)
        return cv2.copyMakeBorder(
            image, pad_y // 2, pad_y - pad_y // 2,
            pad_x // 2, pad_x - pad_x // 2, cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )


def find_regions(detector, frame, args):
    result = detector.predict(frame, conf=args.hook_conf, imgsz=640, verbose=False)[0]
    height, width = frame.shape[:2]
    half_width = width * args.crop_width / 2
    regions = []
    for x1, y1, x2, y2 in result.boxes.xyxy.tolist():
        center_x = (x1 + x2) / 2
        top = max(0, round(y2 - height * 0.04))
        left = max(0, round(center_x - half_width))
        right = min(width, round(center_x + half_width))
        bottom = min(height, round(top + height * args.crop_height))
        if right - left >= 16 and bottom - top >= 16:
            hook = (round(x1), round(y1), round(x2), round(y2))
            regions.append(Region(hook, (left, top, right, bottom)))
    return sorted(regions, key=lambda region: region.hook[0] + region.hook[2])


def load_model(weights, task):
    if not weights.is_file():
        raise RuntimeError(f"가중치 파일을 찾을 수 없습니다. {weights}")
    model = require_yolo()(str(weights))
    if model.task != task:
        raise RuntimeError(f"{weights.name}의 종류는 {model.task}입니다. {task} 모델이 필요합니다.")
    return model


def draw_hook(display, region, number, color):
    x1, y1, x2, y2 = region.hook
    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
    point = (x1, max(18, y1 - 5))
    cv2.putText(display, str(number), point, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(display, str(number), point, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def add_footer(display, lines):
    line_height = 22
    y = display.shape[0]
    display = cv2.copyMakeBorder(display, 0, 6 + len(lines) * line_height, 0, 0,
                                 cv2.BORDER_CONSTANT, value=(28, 28, 28))
    for line in lines:
        y += line_height
        cv2.putText(display, line, (8, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (245, 245, 245), 1, cv2.LINE_AA)
    return display


def read_key(display):
    cv2.imshow(WINDOW, display)
    key = cv2.waitKey(1) & 0xFF
    return ord("q") if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1 else key


def save_crop(folder, crop):
    ok, encoded = cv2.imencode(".jpg", crop)
    if not ok:
        raise OSError("분류용 사진을 JPEG로 변환하지 못했습니다.")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid4().hex}.jpg"
    path.write_bytes(encoded.tobytes())
    return path


def collect(camera, detector, args):
    frozen = None
    regions = []
    saved = {}
    selected = 0
    split = args.split.upper()
    print("Space: 촬영 화면 멈춤/다음 촬영 | Tab: 다음 고리 | 1: 빈 옷걸이 | 2: 옷이 걸림 | Q: 종료")
    print(f"사진 저장 위치 {args.dataset / args.split}")
    while True:
        if frozen is None:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("카메라 영상을 읽지 못했습니다.")
            display = frame
            lines = [f"{split} | Space: capture | Q: quit"]
        else:
            display = frozen.copy()
            for index, region in enumerate(regions):
                color = (0, 225, 255) if index == selected else (190, 190, 190)
                draw_hook(display, region, index + 1, color)
            if regions:
                x1, y1, x2, y2 = regions[selected].bounds
                cv2.rectangle(display, (x1, y1), (x2 - 1, y2 - 1), (0, 225, 255), 2)
                status = f" | saved: {LABELS[saved[selected]]}" if selected in saved else ""
                lines = [f"{split} | Hanger {selected + 1}/{len(regions)}{status}",
                         "1: empty | 2: with clothes | Tab: next | Space: live | Q: quit"]
            else:
                lines = [f"{split} | No hooks detected | Space: live | Q: quit"]

        key = read_key(add_footer(display, lines))
        if key == ord("q"):
            return
        if key == ord(" "):
            if frozen is None:
                frozen = frame
                regions = find_regions(detector, frozen, args)
                saved.clear()
                selected = 0
            else:
                frozen = None
        elif frozen is not None and regions:
            if key == 9:
                selected = (selected + 1) % len(regions)
            elif key in KEY_LABELS and selected not in saved:
                label = KEY_LABELS[key]
                path = save_crop(args.dataset / args.split / label, regions[selected].crop(frozen))
                print(f"{selected + 1}번 {LABELS[label]} 저장 {path}")
                saved[selected] = label


def train(args):
    for split in ("train", "val"):
        for label in (EMPTY, LOADED):
            folder = args.dataset / split / label
            if not any(p.is_file() for ext in ("*.jpg", "*.jpeg", "*.png") for p in folder.glob(ext)):
                raise RuntimeError(f"{folder}에 사진이 없습니다. collect --split {split}으로 예시를 저장하세요.")
    model = require_yolo()(args.weights)
    if model.task != "classify":
        raise RuntimeError("분류 모델을 지정하세요. 예: yolov8n-cls.pt")
    model.train(data=str(args.dataset.resolve()), epochs=args.epochs, imgsz=STATE_IMAGE_SIZE,
                workers=0, project=str(RUNS), name="classify", exist_ok=False)
    print(f"학습된 상태 분류 모델 {model.trainer.best}")


def classify(classifier, frame, regions, min_score):
    if not regions:
        return []
    crops = [region.crop(frame) for region in regions]
    results = classifier.predict(crops, imgsz=STATE_IMAGE_SIZE, verbose=False)
    states = []
    for result in results:
        score = float(result.probs.top1conf)
        label = result.names[result.probs.top1] if score >= min_score else "unknown"
        states.append((label, score))
    return states


def detect(camera, detector, args):
    classifier = load_model(args.state_weights, "classify")
    if set(classifier.names.values()) != {EMPTY, LOADED}:
        raise RuntimeError("상태 분류에는 empty_hanger와 with_clothes를 학습한 모델이 필요합니다.")
    while True:
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("카메라 영상을 읽지 못했습니다.")
        regions = find_regions(detector, frame, args)
        states = classify(classifier, frame, regions, args.min_state_score)
        lines = ["Live states | Q: quit"]
        for index, (region, (label, score)) in enumerate(zip(regions, states), start=1):
            draw_hook(frame, region, index, COLORS[label])
            lines.append(f"{index}. {LABELS[label]} ({score:.2f})")
        if not regions:
            lines.append("No hooks detected")
        if read_key(add_footer(frame, lines)) == ord("q"):
            return


def fraction(value):
    result = float(value)
    if not 0 < result <= 1:
        raise argparse.ArgumentTypeError("expected a number greater than 0 and at most 1")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    camera_options = argparse.ArgumentParser(add_help=False)
    camera_options.add_argument("--camera", type=int, default=0)
    camera_options.add_argument("--hook-weights", type=Path, default=HOOK_WEIGHTS)
    camera_options.add_argument("--hook-conf", type=fraction, default=0.25)
    camera_options.add_argument("--crop-width", type=fraction, default=0.28, help="fraction of frame width")
    camera_options.add_argument("--crop-height", type=fraction, default=0.60, help="fraction of frame height")

    collection = subcommands.add_parser("collect", parents=[camera_options], help="save labelled crops")
    collection.add_argument("--split", choices=("train", "val"), required=True)
    collection.add_argument("--dataset", type=Path, default=DATASET)
    training = subcommands.add_parser("train", help="fine-tune the state classifier")
    training.add_argument("--dataset", type=Path, default=DATASET)
    training.add_argument("--weights", default="yolov8n-cls.pt")
    training.add_argument("--epochs", type=int, default=30)
    detection = subcommands.add_parser("detect", parents=[camera_options], help="detect hooks and classify each crop")
    detection.add_argument("--state-weights", type=Path, required=True)
    detection.add_argument("--min-state-score", type=fraction, default=0.70)

    args = parser.parse_args()
    try:
        if args.command == "train":
            train(args)
            return
        detector = load_model(args.hook_weights, "detect")
        camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        try:
            if not camera.isOpened():
                raise RuntimeError("카메라를 열 수 없습니다. 다른 카메라 번호는 --camera 1로 선택할 수 있습니다.")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            action = collect if args.command == "collect" else detect
            action(camera, detector, args)
        finally:
            camera.release()
            cv2.destroyAllWindows()
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"오류: {exc}\n")


if __name__ == "__main__":
    main()
