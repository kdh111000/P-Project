"""사용자 요청과 날씨를 코디 검색에 사용할 의류 조건으로 변환한다."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from .schemas import GarmentSpec, OutfitPlan, OutfitPlanInput, UserQuery
from .prompts import OUTFIT_PLAN_PROMPT
from .weather import WeatherInfo, WeatherLookupError, get_weather

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def generate_outfit_plan(
    query: str,
    weather: WeatherInfo,
    *,
    client: OpenAI | None = None,
    model: str | None = None,
    intent: str | None = None,
    selected_clothing: dict[str, str] | None = None,
) -> OutfitPlan:
    """조회된 날씨와 사용자 요청을 OpenAI에 보내 의류 카테고리별 조건을 반환한다."""
    payload = OutfitPlanInput(
        query=query, weather=weather, intent=intent, selected_clothing=selected_clothing
    )
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    model_name = (
        model
        or os.getenv("OPENAI_RECOMMENDATION_MODEL", "").strip()
        or os.getenv("OPENAI_VISION_MODEL", "").strip()
        or "gpt-4.1-mini"
    )
    ai = client if client is not None else OpenAI(timeout=30.0, max_retries=2)
    response = ai.responses.parse(
        model=model_name,
        instructions=OUTFIT_PLAN_PROMPT,
        input=json.dumps(payload.model_dump(exclude_none=True), ensure_ascii=False),
        text_format=OutfitPlan,
        max_output_tokens=1200,
        store=False,
    )
    result = response.output_parsed
    if response.status != "completed" or result is None:
        raise RuntimeError("OpenAI에서 완성된 코디 조건을 받지 못했습니다.")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="날씨를 조회하고 AI 코디 결과를 JSON으로 출력합니다.")
    parser.add_argument("--location", default="서울", help="날씨를 조회할 지역명 (기본값: 서울)")
    parser.add_argument("--country-code", help="지역을 한정할 국가 코드 (예: KR)")
    parser.add_argument("--query", default="출근할 때 입을 단정한 코디", help="코디 요청 문장")
    parser.add_argument("--at", type=datetime.fromisoformat, help="시간대가 포함된 ISO 8601 예보 시각")
    args = parser.parse_args(argv)

    query = args.query.strip()
    if not 1 <= len(query) <= 2000:
        parser.error("코디 요청은 1~2000자로 입력해 주세요.")

    try:
        weather = get_weather(args.location, at=args.at, country_code=args.country_code)
    except ValueError as exc:
        parser.error(str(exc))
    except WeatherLookupError as exc:
        parser.exit(1, f"날씨 조회 실패. {exc}\n")

    try:
        plan = generate_outfit_plan(query, weather)
    except OpenAIError as exc:
        parser.exit(1, f"OpenAI 요청 실패 ({type(exc).__name__}). API 설정과 연결을 확인해 주세요.\n")
    except (RuntimeError, ValidationError):
        parser.exit(1, "OpenAI에서 유효한 코디 결과를 받지 못했습니다.\n")

    result = {
        "location": args.location.strip(),
        "query": query,
        "at": args.at.isoformat() if args.at is not None else None,
        "weather": weather.model_dump(),
        "outfit": plan.model_dump(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
