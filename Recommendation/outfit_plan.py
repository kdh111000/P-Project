"""사용자 요청과 날씨를 코디 검색에 사용할 의류 조건으로 변환한다."""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from DataBase.Scripts.clients import get_supabase_client
from .clothing_search import OutfitWithImages, SelectedClothingImage, search_clothes
from .prompts import OUTFIT_PLAN_PROMPT
from .schemas import GarmentSpec, OutfitPlan, OutfitPlanInput, UserQuery

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


def generate_outfit_with_images(
    query: str,
    weather: WeatherInfo,
    *,
    user_id: str | None = None,
    top_k: int = 3,
    selected_clothing_id: str | UUID | None = None,
) -> OutfitWithImages:
    """LLM 코디 조건과 종류별 보유 의류 검색 결과를 함께 반환한다."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    try:
        owner = str(UUID(str(user_id or os.getenv("IMPORT_USER_ID", ""))))
    except ValueError:
        raise ValueError("사용자 UUID 또는 .env의 IMPORT_USER_ID를 확인해 주세요.") from None
    db = get_supabase_client()
    selected = None
    context = None
    if selected_clothing_id is not None:
        clothing_id = str(UUID(str(selected_clothing_id)))
        rows = (
            db.table("clothes").select("*").eq("user_id", owner)
            .eq("id", clothing_id).limit(1).execute().data
        )
        if not rows:
            raise ValueError("선택한 옷을 사용자 옷장에서 찾을 수 없습니다.")
        row = rows[0]
        selected = SelectedClothingImage(id=row["id"], name=row["name"], category=row["category"])
        context = {"name": selected.name, "category": selected.category}
        for key in ("description", "color", "material"):
            if row.get(key):
                context[key] = str(row[key])
        if row.get("image_bucket") and row.get("image_path"):
            try:
                signed = db.storage.from_(row["image_bucket"]).create_signed_url(row["image_path"], 600)
                selected.photo_url = signed["signedURL"]
            except Exception:
                logging.warning("선택한 옷의 사진 URL 생성 실패: %s", clothing_id)
    plan = generate_outfit_plan(query, weather, selected_clothing=context)
    matches = search_clothes(
        plan, user_id=owner, top_k=top_k, db=db,
        excluded_category=selected.category if selected else None,
    )
    return OutfitWithImages(plan=plan, matches=matches, selected_clothing=selected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="날씨를 조회하고 AI 코디 결과를 JSON으로 출력합니다.")
    parser.add_argument("--location", default="서울", help="날씨를 조회할 지역명 (기본값: 서울)")
    parser.add_argument("--country-code", help="지역을 한정할 국가 코드 (예: KR)")
    parser.add_argument("--query", default="출근할 때 입을 단정한 코디", help="코디 요청 문장")
    parser.add_argument("--at", type=datetime.fromisoformat, help="시간대가 포함된 ISO 8601 예보 시각")
    parser.add_argument("--user-id", help="보유 의류 사용자 UUID · 생략 시 IMPORT_USER_ID 사용")
    parser.add_argument("--top-k", type=int, choices=range(1, 11), default=3)
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
        recommendation = generate_outfit_with_images(
            query, weather, user_id=args.user_id, top_k=args.top_k
        )
    except OpenAIError as exc:
        parser.exit(1, f"OpenAI 요청 실패 ({type(exc).__name__}). API 설정과 연결을 확인해 주세요.\n")
    except Exception as exc:
        parser.exit(1, f"코디 생성 또는 의류 검색 실패 ({type(exc).__name__}). {exc}\n")

    result = {
        "location": args.location.strip(),
        "query": query,
        "at": args.at.isoformat() if args.at is not None else None,
        "weather": weather.model_dump(),
        "outfit": recommendation.plan.model_dump(),
        "matches": recommendation.matches.model_dump(mode="json"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
