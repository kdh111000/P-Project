"""전처리된 의류 이미지를 LLM에 보내 속성을 추출한다.

파이프라인과 독립된 모듈이라 호출하는 쪽에서 필요할 때만 가져다 쓰면 된다.
실행하려면 OPENAI_API_KEY 환경변수가 있어야 한다.

    results = Pipeline().run_garment(["front.jpg", "back.jpg"])
    attrs = extract_attributes(results)
"""

import base64
import io
import json

from . import config
from .pipeline import merge_category

INSTRUCTIONS = (
    "의류 사진을 보고 속성을 정리한다. 입력은 같은 옷 한 벌을 여러 방향에서 찍은 사진이다. "
    "옷을 잡고 있는 손이나 남아 있는 배경은 무시한다. "
    "모든 값은 한국어로 짧게 쓰고, 사진으로 판단하기 어려운 항목은 '알 수 없음'으로 적는다."
)


def _to_data_url(image, max_side=config.LLM_IMAGE_SIZE):
    """긴 변을 max_side로 줄여 JPEG data URL로 만든다. 이미지가 클수록 토큰 비용이 늘어난다."""
    img = image.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_content(results, category=None):
    """Responses API의 user content를 만든다. 사진마다 어느 방향인지 알려 주는 문장을 앞에 붙인다."""
    content = []
    if category:
        content.append({"type": "input_text",
                        "text": f"종류: {config.KOREAN.get(category, category)}"})
    for i, r in enumerate(results, start=1):
        label = config.KOREAN[r.view] if r.view else f"사진 {i}"
        content.append({"type": "input_text", "text": f"[{label}]"})
        content.append({"type": "input_image", "image_url": _to_data_url(r.image), "detail": "auto"})
    return content


def _schema(attributes):
    return {
        "type": "object",
        "properties": {k: {"type": "string", "description": v} for k, v in attributes.items()},
        "required": list(attributes),
        "additionalProperties": False,
    }


def extract_attributes(results, model=config.LLM_MODEL, attributes=config.ATTRIBUTES, client=None):
    """옷 한 벌의 처리 결과 묶음을 한 번의 요청으로 보내고 속성을 dict로 받는다.

    방향별로 따로 물으면 답이 엇갈릴 수 있고, 뒷면에만 있는 무늬처럼 여러 장을 같이 봐야
    알 수 있는 정보도 있어서 한 요청에 묶는다.
    """
    if client is None:
        from openai import OpenAI
        client = OpenAI()

    category = merge_category(results)
    response = client.responses.create(
        model=model,
        instructions=INSTRUCTIONS,
        input=[{"role": "user", "content": build_content(results, category)}],
        text={"format": {
            "type": "json_schema",
            "name": "garment_attributes",
            "schema": _schema(attributes),
            "strict": True,
        }},
    )
    return {"category": category, **json.loads(response.output_text)}
