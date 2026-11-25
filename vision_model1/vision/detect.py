"""사진에서 등록할 옷을 찾고, 등록에 쓸 만한 상태로 찍혔는지 판정한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .dataset import load_image


@dataclass
class Detection:
    box: tuple[int, int, int, int]   # x1, y1, x2, y2 (원본 픽셀 좌표)
    conf: float
    label: str
    mask: np.ndarray | None = None   # 원본과 같은 크기의 bool 배열. 분할 모델일 때만 있다


class GarmentDetector:
    """Ultralytics 가중치를 감싼 탐지기.

    YOLOE(텍스트 프롬프트)와 일반 YOLO 가중치를 모두 받는다. 어느 쪽이든 detect()의
    반환 형식은 같아서, 가중치를 바꿔도 뒤 단계 코드는 그대로 쓸 수 있다.
    """

    def __init__(self, weights=config.DETECT_WEIGHTS, prompts=config.DETECT_PROMPTS,
                 text_prompt=None, device=None):
        from ultralytics import YOLO, YOLOE

        if text_prompt is None:
            text_prompt = Path(str(weights)).name.lower().startswith("yoloe")
        if text_prompt:
            self.model = YOLOE(weights)
            self.model.set_classes(prompts, self.model.get_text_pe(prompts))
        else:
            self.model = YOLO(weights)
        self.device = device

    def detect(self, image, conf=config.DETECT_CONF):
        """신뢰도 conf 이상인 옷 후보를 모두 돌려준다."""
        img = load_image(image)
        # retina_masks=True 여야 마스크가 원본 해상도로 나온다.
        # 프롬프트가 여러 개라 같은 옷이 다른 이름으로 겹쳐 잡히므로 클래스 구분 없이 NMS를 건다.
        result = self.model.predict(img, conf=conf, retina_masks=True, agnostic_nms=True,
                                    device=self.device, verbose=False)[0]
        boxes = result.boxes
        masks = None
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy() > 0.5

        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = (int(round(v)) for v in boxes.xyxy[i].tolist())
            detections.append(Detection(
                box=(x1, y1, x2, y2),
                conf=float(boxes.conf[i]),
                label=result.names[int(boxes.cls[i])],
                mask=masks[i] if masks is not None else None,
            ))
        return detections

    def detect_one(self, image, pick=config.PICK):
        """등록 대상 한 벌만 고른다. 후보가 없으면 None."""
        img = load_image(image)
        return pick_target(self.detect(img), img.size, pick)


def _area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def pick_target(detections, image_size, how=config.PICK):
    """여러 후보 중 등록 대상을 고른다.

    사용자가 옷을 몸 앞에 들고 찍으면 입고 있는 옷도 함께 잡힐 수 있다. 들고 있는 옷이
    카메라에 더 가깝고 크게 찍히므로 기본값은 가장 큰 박스다.
    """
    if not detections:
        return None
    if how == "largest":
        return max(detections, key=lambda d: _area(d.box))
    if how == "conf":
        return max(detections, key=lambda d: d.conf)
    if how == "center":
        w, h = image_size

        def dist(d):
            x1, y1, x2, y2 = d.box
            return ((x1 + x2) / 2 - w / 2) ** 2 + ((y1 + y2) / 2 - h / 2) ** 2

        return min(detections, key=dist)
    raise ValueError(f"알 수 없는 선택 기준: {how}")


def check_ready(detection, image_size):
    """등록에 써도 되는 사진인지 판정한다. (통과 여부, 사유)를 돌려준다.

    사유 문자열은 화면 안내 문구로 그대로 쓸 수 있게 짧게 적었다.
    """
    if detection is None:
        return False, "옷을 찾지 못했습니다"

    w, h = image_size
    x1, y1, x2, y2 = detection.box
    if detection.conf < config.READY_MIN_CONF:
        return False, "옷이 또렷하게 보이지 않습니다"

    mx, my = w * config.READY_EDGE_MARGIN, h * config.READY_EDGE_MARGIN
    if x1 <= mx or y1 <= my or x2 >= w - mx or y2 >= h - my:
        return False, "옷이 화면 밖으로 나갔습니다"

    if _area(detection.box) / (w * h) < config.READY_MIN_AREA:
        return False, "옷을 카메라에 더 가까이 대 주세요"

    return True, "ok"
