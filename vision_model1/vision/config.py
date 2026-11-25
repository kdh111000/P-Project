"""경로, 라벨, 기준값 모음. 환경에 따라 바뀌는 값은 여기서만 고친다."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"      # 촬영 원본
CROP_DIR = ROOT / "data" / "crops"   # 탐지·크롭을 거친 학습용 이미지
PUBLIC_DIR = ROOT / "data" / "public"  # 공개 데이터(DeepFashion2)에서 만든 학습용 이미지
WEIGHT_DIR = ROOT / "weights"

# 폴더 이름과 모델 출력에 쓰는 라벨. 순서가 곧 클래스 인덱스다.
CATEGORIES = ["top", "bottom", "outer"]
VIEWS = ["front", "back"]
KOREAN = {
    "top": "상의", "bottom": "하의", "outer": "외투",
    "front": "앞면", "back": "뒷면",
}

# ---- 탐지 ----
# yoloe-*.pt 는 텍스트 프롬프트로 대상을 지정하는 사전학습 모델이라 추가 학습 없이 쓸 수 있다.
# 직접 학습한 YOLO 가중치를 쓸 때는 경로만 바꾸면 된다.
DETECT_WEIGHTS = "yoloe-11s-seg.pt"
DETECT_PROMPTS = [
    "shirt", "t-shirt", "sweater", "hoodie",
    "jacket", "coat", "pants", "jeans", "skirt",
]
DETECT_CONF = 0.05       # 후보로 남길 최소 신뢰도. 사전학습 모델은 점수가 낮게 나와서 작게 잡았다.
PICK = "largest"         # 여러 벌이 잡혔을 때 고르는 기준: largest | center | conf

# 촬영 상태 판정 기준. 카메라 화각과 설치 위치에 맞춰 조정한다.
READY_MIN_CONF = 0.10    # 이보다 낮으면 옷이 아닐 가능성이 높다고 본다
READY_MIN_AREA = 0.10    # 화면 대비 박스 넓이. 너무 멀리서 든 경우를 걸러낸다
READY_EDGE_MARGIN = 0.01  # 박스가 화면 끝에 닿으면 옷이 잘린 것으로 본다

# ---- 등록 진행 (카메라 스트림) ----
READY_HOLD = 1.0         # 이 시간(초) 동안 계속 통과해야 저장한다. 옷을 펴는 도중에 찍히는 것을 막는다
CAPTURE_COOLDOWN = 2.0   # 저장 후 다음 방향을 받기까지 기다리는 시간(초). 옷을 돌릴 여유
# 방향 확인은 느슨하게 건다. 무지 티셔츠나 바지처럼 앞뒤가 비슷한 옷은 모델도 자주 틀리기 때문에,
# 다른 방향이라고 강하게 확신할 때만 다시 요청하고 그마저도 횟수를 넘기면 안내 순서를 믿는다.
VIEW_REJECT_CONF = 0.8
VIEW_MAX_REJECTS = 2

# ---- 크롭 ----
CROP_PAD = 0.03          # 박스 바깥으로 남기는 여백(박스 긴 변 대비)
BG_COLOR = (255, 255, 255)

# ---- 분류 ----
ARCH = "resnet18"
IMG_SIZE = 224

# ---- LLM ----
LLM_MODEL = os.getenv("FITMATE_LLM_MODEL", "gpt-5.6")
LLM_IMAGE_SIZE = 768     # 전송 전 긴 변을 이 크기로 줄인다

# LLM에 요청할 속성. 키 이름은 DB 컬럼과 맞춘다.
ATTRIBUTES = {
    "color": "가장 넓은 면적을 차지하는 색",
    "sub_color": "두 번째로 눈에 띄는 색. 없으면 빈 문자열",
    "pattern": "무늬 (무지, 스트라이프, 체크, 프린트 등)",
    "material": "사진으로 추정한 소재",
    "thickness": "두께감 (얇음, 보통, 두꺼움)",
    "season": "입기 적당한 계절",
    "style": "스타일 (캐주얼, 포멀, 스포티 등)",
    "description": "한 문장 요약",
}
