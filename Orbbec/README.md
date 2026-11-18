# Orbbec 옷장 카메라 (옷 입출고 기록)

옷장 위에 단 Orbbec Astra(깊이 카메라)가 문이 열린 동안 사용자를 녹화하고,
백엔드가 녹화본에서 옷을 찾아 어떤 옷을 꺼냈는지·넣었는지 기록합니다.

| 파일 | 역할 |
|---|---|
| `doorcam.py` | 카메라 녹화(문 열림 감지, 깊이로 사람 감지, 녹화) + 백엔드 판정(`--worker`) |
| `detect.py` | 옷 찾기: OWLv2(글자로 탐지) → SAM(픽셀 분할) → 깊이로 사용자와 함께 있는 옷만 남김 |
| `test_doorcam.py` | 가짜 프레임으로 세션 분리·녹화·판정 로직 테스트 |

## 동작
1. **카메라**: 컬러 밝기로 문 열림 판단(닫힌 문은 빛을 막음). 깊이 레이저는 문이 열린 동안만 켬(Astra 발열 대책).
2. **녹화**: 문이 열린 동안 `doorcam/<시각>/colour/*.jpg`, `depth/*.png`(16-bit mm)를 10 fps로 저장.
3. **판정**(`--worker`): 사람이 잡힌 처음·마지막 10장 중 가장 선명한 장면 → `detect.find`로 옷 조각 추출 →
   조각마다 Qwen3-VL이 등록된 옷 중 몇 번인지 판별(FashionCLIP으로 후보 5벌) → 꺼냄 = 닫을 때만 있는 옷, 넣음 = 열 때만 있는 옷.

## 실행
```bash
python doorcam.py --closet <옷장코드> --show   # 카메라(실시간 창: b = 빈 장면 기억, q = 종료)
python doorcam.py --worker                     # 백엔드 판정
python doorcam.py --stop                       # 카메라 정상 종료(레이저 끔). 강제 종료하면 레이저가 켜진 채 남을 수 있음
python test_doorcam.py
```

## 준비물
- Astra (USB 2BC5:0404) + Orbbec OpenNI2 SDK 폴더(`OPENNI2_DIR`, 기본 `~/Downloads/Orbbec/OpenNI2/sdk/libs`), `pip install openni`
- `transformers`, `torch`, `torchvision`, `opencv-python`, `pillow` (모델: `google/owlv2-base-patch16-ensemble`, `facebook/sam-vit-base`)
- 판정 단계는 Fitmate 옷장 서버 모듈(`web.py`, `extract.py`, `retrieve.py`)과 로컬 Qwen3-VL 서버(llama.cpp, 127.0.0.1:8090)를 씁니다. 이 브랜치에는 포함돼 있지 않습니다.

## 확인된 것 / 한계
- 저장 사진 기준 비교: SegFormer는 손에 든 옷을 티셔츠와 합침, Grounding DINO는 흔들린 든 옷을 놓침, OWLv2+SAM이 둘 다 잡음.
- 등록 사진 합성 4건(든 바지 무시, 갈색 재킷 꺼냄, 군복 꺼냄, 갈색 재킷 넣음) 모두 정답. 실제 옷장 장착 상태 검증은 아직.
- CPU 기준 기록당 1분 안팎. 기준값(`DARK`, `CLOSER_MM`, `MIN_SCORE` 등)은 장착 후 조정 필요.
