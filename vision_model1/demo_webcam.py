"""웹캠으로 옷 등록 흐름을 확인한다.

    python demo_webcam.py                              # 기본 카메라
    python demo_webcam.py --source 1                   # 다른 카메라
    python demo_webcam.py --save-raw data/raw/01_top   # 저장된 원본을 학습 데이터 폴더에 모으기

분류 가중치(weights/view.pt, weights/category.pt)가 없으면 해당 단계는 건너뛴다.
키: q 종료, r 처음부터 다시
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vision import config
from vision.pipeline import Pipeline
from vision.session import RegistrationSession

WINDOW = "Fitmate - garment registration"
# OpenCV의 putText는 한글을 그리지 못해서 PIL로 그린다. 운영체제별 기본 한글 글꼴 위치.
FONT_PATHS = [
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]
SHORT = {"front": "앞", "back": "뒤"}
BOX_COLORS = {"holding": (40, 200, 90), "captured": (40, 200, 90), "done": (40, 200, 90),
              "wrong_view": (230, 60, 60)}


def load_font(size):
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return None


def to_pil(frame_bgr):
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def draw_overlay(frame_bgr, state, font, small_font, category=None):
    """안내 문구, 진행 상황, 탐지 박스를 프레임 위에 그린다. 종류가 정해졌으면 함께 표시한다."""
    img = to_pil(frame_bgr)
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    if state.box:
        color = BOX_COLORS.get(state.status, (255, 150, 30))
        d.rectangle(state.box, outline=color + (255,), width=max(3, w // 200))

    bar_h = int(h * 0.19)
    d.rectangle((0, 0, w, bar_h), fill=(0, 0, 0, 170))
    progress = " ".join(("●" if v in state.captured else "○") + SHORT[v] for v in config.VIEWS)
    if font is None:
        # 한글 글꼴을 못 찾은 환경. 상태 코드만 표시하고 문구는 터미널에 출력한다
        cv2_img = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        cv2.putText(cv2_img, f"{state.status} {len(state.captured)}/{len(config.VIEWS)}",
                    (20, bar_h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return cv2_img

    d.text((w * 0.03, bar_h * 0.10), state.message, font=font, fill="white")
    hint = f"종류: {config.KOREAN[category]}" if category else state.hint
    d.text((w * 0.03, bar_h * 0.62), hint, font=small_font, fill=(255, 210, 120))
    tw = d.textlength(progress, font=small_font)
    d.text((w * 0.97 - tw, bar_h * 0.62), progress, font=small_font, fill="white")
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def run(source=0, weight_dir=config.WEIGHT_DIR, save_raw=None, save_crop=None):
    """등록을 한 번 이상 진행하고, 마지막 세션을 돌려준다."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"카메라(또는 영상)를 열 수 없습니다: {source}")

    pipeline = Pipeline(weight_dir)
    session = RegistrationSession(pipeline)
    font = small_font = None
    last_message = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if font is None:
            font, small_font = load_font(frame.shape[0] // 17), load_font(frame.shape[0] // 26)

        saved_before = len(session.results)
        state = session.step(to_pil(frame))

        if len(session.results) > saved_before:
            view = session.results[-1].view
            stamp = time.strftime("%Y%m%d_%H%M%S")
            if save_raw:
                out = Path(save_raw) / view
                out.mkdir(parents=True, exist_ok=True)
                session.frames[-1].save(out / f"{stamp}.jpg", quality=95)
            if save_crop:
                out = Path(save_crop)
                out.mkdir(parents=True, exist_ok=True)
                session.results[-1].image.save(out / f"{stamp}_{view}.jpg", quality=95)

        if state.message != last_message:
            print(f"[{state.status}] {state.message}")
            last_message = state.message
        if state.status == "done" and len(session.results) > saved_before:
            summary = "저장된 방향: " + ", ".join(config.KOREAN[v] for v in state.captured)
            if session.category:
                summary += f" / 종류: {config.KOREAN[session.category]}"
            print(summary)

        cv2.imshow(WINDOW, draw_overlay(frame, state, font, small_font, session.category))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            session = RegistrationSession(pipeline)

    cap.release()
    cv2.destroyAllWindows()
    return session


def main():
    p = argparse.ArgumentParser(description="웹캠으로 옷 등록 흐름 확인")
    p.add_argument("--source", default="0", help="카메라 번호 또는 영상 파일 경로")
    p.add_argument("--weights", default=str(config.WEIGHT_DIR))
    p.add_argument("--save-raw", default=None,
                   help="저장된 원본 프레임을 <폴더>/<방향>/ 아래에 남긴다. 예) data/raw/01_top")
    p.add_argument("--save-crop", default=None, help="크롭된 이미지를 남길 폴더")
    args = p.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run(source, args.weights, args.save_raw, args.save_crop)


if __name__ == "__main__":
    main()
