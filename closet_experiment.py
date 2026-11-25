"""Interactive, example-based closet vision experiment (OpenCV only)."""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np

"""일단 구역 하나로 박스쳐서 1. 옷걸이 없음, 2. 빈 옷걸이, 3. 옷 있음 예시를 저장해서
잘 맞추는지 확인하는 식으로 수행 -- 추후에 박스여러개로 각 구역별로 인식하게 해야됨"""

ROOT = Path(__file__).resolve().parent
STATES = ("no_hanger", "empty_hanger", "with_clothes")
WINDOW = "Closet experiment"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
selection = {"start": None, "end": None, "dragging": False}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        selection.update(start=(x, y), end=(x, y), dragging=True)
    elif event == cv2.EVENT_MOUSEMOVE and selection["dragging"]:
        selection["end"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and selection["dragging"]:
        selection.update(end=(x, y), dragging=False)


def selected_crop(frame):
    start, end = selection["start"], selection["end"]
    if start is None or end is None:
        return None
    height, width = frame.shape[:2]
    x1, x2 = sorted((max(0, min(start[0], width)), max(0, min(end[0], width))))
    y1, y2 = sorted((max(0, min(start[1], height)), max(0, min(end[1], height))))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return frame[y1:y2, x1:x2]


def feature(image):
    # A deliberately simple baseline: colour distribution and coarse image layout.
    image = cv2.resize(image, (64, 128))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
    hist /= max(float(hist.sum()), 1.0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    layout = cv2.resize(gray, (8, 16)).astype(np.float32).flatten() / 255.0
    return hist, layout


def similarity(a, b):
    color = float(np.minimum(a[0], b[0]).sum())
    layout = 1.0 - float(np.abs(a[1] - b[1]).mean())
    return max(0.0, min(1.0, 0.65 * color + 0.35 * layout))


def read_image(path):
    # imdecode supports non-ASCII file paths on Windows.
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def load_examples(folder):
    result = []
    if not folder.exists():
        return result
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image = read_image(path)
            if image is not None:
                result.append((path, feature(image)))
    return result


def save_image(folder, image):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{time.time_ns()}.jpg"
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok:
        raise OSError("Could not encode image")
    encoded.tofile(str(path))
    return path


def ascii_label(value):
    return str(value).encode("ascii", "replace").decode("ascii")


def draw_info_panel(image, lines):
    """Wrap the on-screen text to the frame width and draw it once on black."""
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48 if width >= 600 else 0.36
    thickness = 1
    margin = 8
    available_width = max(1, width - margin * 2)
    wrapped = []
    for line in lines:
        current = ""
        for word in line.split():
            candidate = f"{current} {word}" if current else word
            if current and cv2.getTextSize(candidate, font, scale, thickness)[0][0] > available_width:
                wrapped.append(current)
                current = word
            else:
                current = candidate
        if current:
            wrapped.append(current)
    text_height = cv2.getTextSize("Ag", font, scale, thickness)[0][1]
    line_height = text_height + 10
    panel_height = min(height, margin * 2 + len(wrapped) * line_height)
    cv2.rectangle(image, (0, 0), (width - 1, panel_height - 1), (0, 0, 0), -1)
    for index, line in enumerate(wrapped):
        y = margin + text_height + index * line_height
        if y >= height:
            break
        cv2.putText(image, line, (margin, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera device number")
    parser.add_argument("--examples", type=Path, default=ROOT / "closet_examples")
    parser.add_argument("--references", type=Path, default=ROOT / "clothing_references")
    parser.add_argument("--save-id", default="test_item", help="folder name for side-view samples")
    args = parser.parse_args()
    if args.save_id in (".", "..") or any(c in args.save_id for c in '/\\:'):
        parser.error("--save-id must be a single folder name")

    examples = load_examples(args.examples)
    references = load_examples(args.references)
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        print("카메라를 열 수 없습니다. --camera 1 등 다른 번호를 확인하세요.")
        return
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print("마우스로 한 홈 영역을 드래그하세요. 1: 옷걸이 없음, 2: 빈 옷걸이, 3: 옷 있음 예시 저장")
    print("s: 현재 구역을 의류 기준 사진으로 저장, r: 파일 다시 읽기, c: 구역 초기화, q: 종료")
    print("h: 화면 위 안내 표시 켜기/끄기")
    print(f"상태 예시 폴더: {args.examples.resolve()}")
    print(f"의류 기준 사진 폴더: {args.references.resolve()}")

    show_info = True
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("카메라 영상을 읽지 못했습니다.")
                break
            crop = selected_crop(frame)
            display = frame.copy()
            lines = ["Drag: select ROI | 1/2/3: save | R: reload | C: clear | H: hide | Q: quit"]
            if crop is not None:
                f = feature(crop)
                state_scores = {}
                for path, ref in examples:
                    label = path.relative_to(args.examples).parts[0]
                    if label in STATES:
                        state_scores[label] = max(state_scores.get(label, 0), similarity(f, ref))
                if all(label in state_scores for label in STATES):
                    scores = sorted(state_scores.items(), key=lambda x: x[1], reverse=True)
                    lines.append("State candidate: " + scores[0][0] + " | " + " ".join(f"{k}:{v:.2f}" for k, v in scores))
                else:
                    missing = ", ".join(label for label in STATES if label not in state_scores)
                    lines.append("Save examples for all 3 states first. Missing: " + missing)
                item_scores = {}
                for path, ref in references:
                    relative = path.relative_to(args.references)
                    # Subfolders represent IDs; files directly under references use their filename.
                    item_id = relative.parts[0] if len(relative.parts) > 1 else path.stem
                    item_scores[item_id] = max(item_scores.get(item_id, 0), similarity(f, ref))
                if item_scores:
                    best = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    lines.append("Item candidates (NOT confirmed): " + "  ".join(f"{ascii_label(k)}:{v:.2f}" for k, v in best))
                else:
                    lines.append("No item references: use S or add images to reference folders.")
                cv2.rectangle(display, selection["start"], selection["end"], (0, 0, 255), 2)
            else:
                lines.append("Select a region with the mouse.")
            if show_info:
                draw_info_panel(display, lines)
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('c'):
                selection.update(start=None, end=None, dragging=False)
            elif key == ord('h'):
                show_info = not show_info
            elif key == ord('r'):
                examples, references = load_examples(args.examples), load_examples(args.references)
                print(f"다시 읽음: 상태 예시 {len(examples)}개, 의류 사진 {len(references)}개")
            elif key in (ord('1'), ord('2'), ord('3'), ord('s')):
                if crop is None:
                    print("먼저 화면에서 구역을 드래그하세요.")
                    continue
                try:
                    if key == ord('s'):
                        path = save_image(args.references / args.save_id, crop)
                        references = load_examples(args.references)
                    else:
                        label = STATES[key - ord('1')]
                        path = save_image(args.examples / label, crop)
                        examples = load_examples(args.examples)
                    print(f"저장: {path}")
                except OSError as exc:
                    print(f"이미지 저장 실패: {exc}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
