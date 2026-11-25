"""옷 한 벌의 등록 과정을 카메라 프레임 단위로 진행한다.

화면에 안내한 방향(앞 → 뒤) 순서대로 한 장씩 모으고, 다 모이면 종류를 분류한다.
호출하는 쪽은 프레임이 들어올 때마다 step()을 부르고, 돌려받은 message를 화면에 띄우면 된다.

    session = RegistrationSession(Pipeline())
    while not session.done:
        state = session.step(frame)      # state.message 를 화면에 표시
    attrs = extract_attributes(session.results)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import config
from .crop import crop_garment
from .dataset import load_image
from .detect import check_ready
from .pipeline import Result, merge_category

PROMPTS = {view: f"옷의 {config.KOREAN[view]}을 보여주세요" for view in config.VIEWS}
DONE_MESSAGE = "촬영이 끝났습니다"


@dataclass
class StepState:
    status: str                  # waiting | holding | captured | wrong_view | done
    message: str                 # 화면에 띄울 안내 문구
    hint: str = ""               # 통과하지 못한 이유
    target: str | None = None    # 지금 찍어야 하는 방향. 끝났으면 None
    box: tuple | None = None     # 탐지 박스. 화면에 표시할 때 쓴다
    captured: tuple = ()         # 지금까지 저장된 방향


class RegistrationSession:
    """등록 한 번(옷 한 벌)에 하나씩 만든다.

    가중치 파일이 없는 분류 단계는 건너뛴다. check_view, use_category 로 직접 끌 수도 있다.
    clock 은 테스트에서 시간을 직접 넘기기 위한 인자다.
    """

    def __init__(self, pipeline, views=config.VIEWS, check_view=True, use_category=True,
                 clock=time.monotonic):
        self.pipeline = pipeline
        self.views = list(views)
        self.check_view = check_view and pipeline.has("view")
        self.use_category = use_category and pipeline.has("category")
        self._clock = clock

        self.results = []        # 저장된 사진의 처리 결과 (안내한 순서)
        self.frames = []         # 저장된 원본 프레임
        self.category = None
        self._ready_since = None
        self._rejects = 0
        self._resume_at = 0.0

    @property
    def target(self):
        return self.views[len(self.results)] if len(self.results) < len(self.views) else None

    @property
    def done(self):
        return self.target is None

    def _state(self, status, message, **extra):
        return StepState(status, message, target=self.target,
                         captured=tuple(r.view for r in self.results), **extra)

    def step(self, frame):
        """프레임 한 장을 처리하고 현재 상태를 돌려준다."""
        if self.done:
            return self._state("done", DONE_MESSAGE)

        target = self.target
        prompt = PROMPTS[target]
        now = self._clock()

        # 직전 사진을 저장한 직후에는 사용자가 옷을 돌릴 시간을 준다.
        # 이 대기가 없으면 같은 자세가 그대로 다음 방향으로 저장된다.
        if now < self._resume_at:
            return self._state("waiting", prompt)

        img = load_image(frame)
        det = self.pipeline.detector.detect_one(img)
        ready, reason = check_ready(det, img.size)
        box = det.box if det else None
        if not ready:
            self._ready_since = None
            return self._state("waiting", prompt, hint=reason, box=box)

        # 한 프레임만 보고 저장하면 옷을 펴는 도중의 장면이 찍힐 수 있어 일정 시간 유지를 요구한다
        if self._ready_since is None:
            self._ready_since = now
        if now - self._ready_since < config.READY_HOLD:
            return self._state("holding", "그대로 잠시 들고 계세요", box=box)
        self._ready_since = None

        crop = crop_garment(img, det)
        scores = None
        if self.check_view:
            seen, scores = self.pipeline.classifier("view").predict(crop)
            wrong = seen != target and scores[seen] >= config.VIEW_REJECT_CONF
            if wrong and self._rejects < config.VIEW_MAX_REJECTS:
                self._rejects += 1
                return self._state("wrong_view", f"{config.KOREAN[seen]}으로 보입니다. {prompt}", box=box)

        # 방향 라벨은 모델 예측이 아니라 안내한 순서를 따른다. 예측 확률은 view_scores 에 남긴다
        self._rejects = 0
        self.results.append(Result(image=crop, ready=True, reason="ok", view=target,
                                   view_scores=scores, box=det.box, conf=det.conf))
        self.frames.append(img)
        self._resume_at = now + config.CAPTURE_COOLDOWN

        if self.done:
            self._classify()
            return self._state("done", DONE_MESSAGE, box=box)
        return self._state("captured", f"저장했습니다. {PROMPTS[self.target]}", box=box)

    def _classify(self):
        """모은 사진 전체로 종류를 정한다. 한 벌이므로 모든 사진에 같은 종류를 기록한다."""
        if not self.use_category:
            return
        classifier = self.pipeline.classifier("category")
        for r in self.results:
            _, r.category_scores = classifier.predict(r.image)
        self.category = merge_category(self.results)
        for r in self.results:
            r.category = self.category
