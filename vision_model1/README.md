# 비전 모델 1 · 의류 등록 전처리

새 옷을 등록할 때 카메라 영상을 LLM에 넘기기 전에 거치는 전처리 단계다.
영상에서 등록할 옷만 찾아 잘라내고, 안내한 방향대로 찍혔는지 확인한 뒤 옷의 종류를 분류한다.

## 처리 흐름
<img width="1080" height="1930" alt="image" src="https://github.com/user-attachments/assets/5545020b-8085-4520-a5bf-0e66c0eda099" />



| 단계 | 하는 일 | 모델 |
|---|---|---|
| 탐지 | 옷을 찾고, 등록에 쓸 만하게 찍혔는지 판정 | YOLOE (사전학습) |
| 크롭 | 박스 영역만 자르고 옷이 아닌 픽셀을 흰색으로 덮음 | 탐지 모델의 마스크 |
| 방향 확인 | 안내한 방향과 다른 면을 보여준 경우를 걸러냄 | ResNet18 (직접 학습) |
| 종류 분류 | 상의 / 하의 / 외투 | ResNet18 (직접 학습) |
| 속성 추출 | 색, 무늬, 소재 등 | OpenAI API |

## 폴더 구조

```
vision_model1/
├── vision/
│   ├── config.py       경로, 라벨, 기준값
│   ├── detect.py       옷 탐지, 촬영 상태 판정
│   ├── crop.py         크롭, 배경 제거
│   ├── classifier.py   분류 모델 정의, 저장/로드, 추론
│   ├── dataset.py      이미지 로딩, 학습용 데이터셋
│   ├── train.py        분류 모델 학습
│   ├── deepfashion2.py DeepFashion2에서 학습용 이미지 만들기
│   ├── pipeline.py     사진 단위 처리
│   ├── session.py      등록 진행 (방향별 촬영 순서 관리)
│   └── llm.py          LLM 속성 추출
├── weights/            학습한 분류 모델
├── demo_webcam.py      웹캠으로 등록 흐름 실행
├── train_colab.ipynb   Colab 학습 노트북
└── requirements.txt
```

`data/`(촬영 사진, 학습용 이미지)는 저장소에 포함하지 않는다.

## 설치

```bash
pip install -r requirements.txt
```

탐지 모델 가중치는 처음 실행할 때 자동으로 내려받는다.

## 실행

웹캠 앞에서 안내에 따라 옷의 앞면과 뒷면을 보여주면, 옷이 제대로 잡힌 상태가 1초가량 유지될 때 저장하고 다음 방향으로 넘어간다.

```bash
python demo_webcam.py
```

| 옵션 | 설명 |
|---|---|
| `--source 1` | 다른 카메라 번호, 또는 영상 파일 경로 |
| `--save-raw data/raw/01_top` | 저장된 원본을 `<폴더>/<방향>/` 에 남김 (학습 데이터 수집용) |
| `--save-crop out/` | 크롭된 이미지를 남김 |

`q` 종료, `r` 처음부터 다시. 분류 가중치가 없으면 방향 확인과 종류 분류는 건너뛴다.

이미 찍어 둔 사진은 다음과 같이 처리한다.

```bash
python -m vision.pipeline run front.jpg back.jpg --save out/
```

## 학습

종류 분류는 두 단계로 학습한다. 공개 데이터로 옷의 종류를 폭넓게 익힌 뒤, 실제 카메라로 찍은 사진으로 이어서 학습한다.
방향 확인은 촬영 사진으로만 학습한다.

### 1. 공개 데이터 (DeepFashion2)

[DeepFashion2](https://github.com/switchablenorms/DeepFashion2)의 validation 세트를 내려받아 압축을 푼 뒤 실행한다.

```bash
python -m vision.deepfashion2 --src <압축을 푼 폴더> --zip      # → data/public/, data/public.zip
python -m vision.train --task category --extra data/public     # → weights/category.pt
```

라벨의 윤곽선대로 옷을 하나씩 잘라 종류별 폴더에 저장한다. 크롭 단계가 내보내는 이미지와 같은 형태다.
13개 종류는 아래와 같이 묶었고, 많이 가려졌거나 일부만 확대된 옷은 제외했다.

| 종류 | DeepFashion2 종류 |
|---|---|
| 상의 | short sleeve top, long sleeve top |
| 하의 | shorts, trousers, skirt |
| 외투 | short sleeve outwear, long sleeve outwear |
| 제외 | vest, sling, 원피스 4종 |

DeepFashion2의 방향 라벨은 정면과 '옆 또는 뒤'로만 나뉘어 있어 방향 확인 학습에는 쓰지 않았다.

### 2. 촬영 사진

촬영 사진은 옷마다 폴더를 만들고 그 안에 방향별 폴더를 둔다. 라벨은 폴더 이름에서 읽는다.

```
data/raw/
├── 01_top/        <옷 번호>_<종류>   종류: top(상의), bottom(하의), outer(외투)
│   ├── front/
│   └── back/
└── 02_bottom/
    ├── front/
    └── back/
```

```bash
python -m vision.pipeline crop                                                # data/raw → data/crops (탐지·크롭)
python -m vision.train --task category --init weights/category.pt --lr 1e-4   # 1단계 가중치에서 이어서 학습
python -m vision.train --task view                                            # 방향 확인 → weights/view.pt
```

학습과 검증은 옷 단위로 나눈다. 같은 옷의 사진이 양쪽에 섞이면 검증 정확도가 실제보다 높게 나오기 때문이다.
촬영 사진이 있으면 검증은 촬영 사진으로만 한다.
Colab에서는 `train_colab.ipynb` 를 위에서부터 실행한다.

## 다른 코드에서 사용

프레임이 들어올 때마다 `step()` 을 호출하고, 돌려받은 문구를 화면에 표시한다.

```python
from vision.pipeline import Pipeline
from vision.session import RegistrationSession
from vision.llm import extract_attributes

session = RegistrationSession(Pipeline())
while not session.done:
    state = session.step(frame)          # frame: PIL 이미지 또는 파일 경로
    show(state.message, state.hint)

attrs = extract_attributes(session.results)   # OPENAI_API_KEY 필요
```

| `state.status` | 의미 |
|---|---|
| `waiting` | 옷을 기다리는 중. `hint` 에 통과하지 못한 이유가 들어 있다 |
| `holding` | 옷이 잡혔고 유지 시간을 채우는 중 |
| `captured` | 한 장 저장됨. `message` 가 다음 방향 안내로 바뀐다 |
| `wrong_view` | 안내한 방향과 다른 면으로 판단되어 다시 요청 |
| `done` | 두 장 모두 저장됨 |

LLM 모델과 추출 항목은 `vision/config.py` 의 `LLM_MODEL`, `ATTRIBUTES` 에서 바꾼다.

## 단계별 비교

각 단계의 기여도를 비교할 수 있도록 `Pipeline.run()` 에서 단계를 끌 수 있다.

| 인자 | 끄면 |
|---|---|
| `use_crop=False` | 탐지·크롭 없이 원본 사진을 그대로 사용 |
| `remove_bg=False` | 박스만 자르고 배경은 남김 |
| `use_category=False` | 종류 분류 생략 |
| `use_view=False` | 방향 분류 생략 |

## 라이선스

탐지에 사용하는 Ultralytics 패키지는 AGPL-3.0 이다.
DeepFashion2는 비상업적 연구 목적으로만 사용할 수 있고 재배포가 금지되어 있어, 데이터와 그로부터 만든 이미지는 저장소에 포함하지 않는다.
`weights/` 의 분류 모델은 DeepFashion2로 학습했으므로 같은 조건에 따라 비상업적 용도로만 사용한다.
