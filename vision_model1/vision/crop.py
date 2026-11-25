"""탐지된 옷 영역만 남기고 배경을 지운다."""

import numpy as np
from PIL import Image

from . import config


def _padded_box(box, image_size, pad):
    w, h = image_size
    x1, y1, x2, y2 = box
    p = int(max(x2 - x1, y2 - y1) * pad)
    return max(0, x1 - p), max(0, y1 - p), min(w, x2 + p), min(h, y2 + p)


def _to_square(image, color):
    """비율을 유지한 채 짧은 변에 여백을 채운다. 분류기 입력에서 옷이 찌그러지지 않게 하기 위함."""
    w, h = image.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), color)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def crop_garment(image, detection, remove_bg=True, pad=config.CROP_PAD, bg_color=config.BG_COLOR):
    """박스 영역을 잘라 정사각형 이미지로 돌려준다.

    remove_bg가 켜져 있고 탐지 결과에 마스크가 있으면 옷이 아닌 픽셀을 bg_color로 덮는다.
    마스크가 없는 탐지기(박스만 출력)를 쓰면 박스 크롭만 한다. 범용 배경 제거 모델은
    사람이 함께 찍힌 사진에서 옷 대신 얼굴을 전경으로 남기는 경우가 있어 쓰지 않았다.
    """
    box = _padded_box(detection.box, image.size, pad)
    patch = image.crop(box)

    if remove_bg and detection.mask is not None:
        x1, y1, x2, y2 = box
        mask = detection.mask[y1:y2, x1:x2]
        # 마스크가 비어 있으면 지울 근거가 없으니 패치를 그대로 둔다
        if mask.any():
            arr = np.asarray(patch).copy()
            arr[~mask] = bg_color
            patch = Image.fromarray(arr)

    return _to_square(patch, bg_color)
