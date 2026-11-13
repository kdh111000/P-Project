"""추천 그래프에서 날씨, LLM, 의류 검색, 정렬과 저장을 실행한다."""

import json
import os
from itertools import product
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI

from DataBase.Scripts.clients import get_chroma_client, get_supabase_client
from .prompts import INTENT_ANALYSIS_PROMPT

from .errors import (
    IntentAnalysisFailed,
    InvalidRecommendationLocation,
    RecommendationAuthRequired,
    RecommendationUnavailable,
    SelectedClothingNotFound,
)
from .outfit_plan import PROJECT_ROOT, generate_outfit_plan
from .schemas import ClothingItem, Outfit, RecommendationResponse
from .state import RecommendationState
from .weather import WeatherLookupError, get_weather

_CLOTHING_COLUMNS = "id,name,category,storage_location,image_bucket,image_path"
_CATEGORIES = ("top", "bottom", "outer")
_HIGH_SIMILARITY = 0.85


def _user_db(state: RecommendationState):
    token = state.get("access_token")
    if token is None:
        raise RecommendationAuthRequired("의류 조회에는 사용자 인증이 필요합니다.")
    db = get_supabase_client()
    authorization = f"Bearer {token.get_secret_value()}"
    db.options.headers["Authorization"] = authorization
    db.postgrest.auth(token.get_secret_value())
    return db


def _clothing_context(state: RecommendationState) -> dict[str, str] | None:
    selected = state.get("selected_clothing")
    if selected is None:
        return None
    item = selected["item"]
    context = {"name": item.name, "category": item.category}
    if selected.get("description"):
        context["description"] = selected["description"]
    return context


def load_selected_clothing(state: RecommendationState) -> dict:
    db = _user_db(state)
    rows = (
        db.table("clothes").select(f"{_CLOTHING_COLUMNS},description")
        .eq("user_id", str(state["user_id"]))
        .eq("id", str(state["selected_clothing_id"]))
        .limit(1).execute().data
    )
    if not rows:
        raise SelectedClothingNotFound("선택한 옷을 사용자 옷장에서 찾을 수 없습니다.")
    row = rows[0]
    return {"selected_clothing": {"item": ClothingItem.model_validate(row), "description": row.get("description")}}


def fetch_weather(state: RecommendationState) -> dict:
    location = state.get("location")
    try:
        weather = get_weather(location) if location else get_weather("Seoul", country_code="KR")
    except ValueError as exc:
        raise InvalidRecommendationLocation(str(exc)) from exc
    except WeatherLookupError as exc:
        raise RecommendationUnavailable(str(exc)) from exc
    return {"weather": weather}


def _recommendation_model() -> str:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return (
        os.getenv("OPENAI_RECOMMENDATION_MODEL", "").strip()
        or os.getenv("OPENAI_VISION_MODEL", "").strip()
        or "gpt-4.1-mini"
    )


def analyze_intent(state: RecommendationState) -> dict:
    model = _recommendation_model()
    payload = {
        "query": state["query"],
        "weather": state["weather"].model_dump(exclude_none=True),
        "selected_clothing": _clothing_context(state),
    }
    with OpenAI(timeout=30, max_retries=2) as ai:
        response = ai.responses.create(
            model=model,
            instructions=INTENT_ANALYSIS_PROMPT,
            input=json.dumps(payload, ensure_ascii=False),
            max_output_tokens=400,
            store=False,
        )
    if response.status != "completed" or not response.output_text.strip():
        raise IntentAnalysisFailed("코디 요청의 의도를 분석하지 못했습니다.")
    return {"intent": response.output_text.strip()}


def generate_plan(state: RecommendationState) -> dict:
    return {"outfit_plan": generate_outfit_plan(
        state["query"], state["weather"],
        intent=state["intent"], selected_clothing=_clothing_context(state),
    )}


def retrieve_candidates(state: RecommendationState) -> dict:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    name = os.getenv("CHROMA_COLLECTION", "clothes_descriptions_v1")
    collection = get_chroma_client().get_collection(name, embedding_function=None)
    metadata = collection.metadata or {}
    model = metadata.get("embedding_model")
    dimensions = metadata.get("embedding_dimensions")
    if not isinstance(model, str) or not isinstance(dimensions, int) or metadata.get("hnsw:space") != "cosine":
        raise RecommendationUnavailable("의류 검색 컬렉션의 임베딩 설정을 확인해 주세요.")

    selected = state.get("selected_clothing")
    fixed_category = selected["item"].category if selected else None
    plan = state["outfit_plan"]
    searches = [(category, getattr(plan, category)) for category in _CATEGORIES if category != fixed_category]
    searches = [(category, spec) for category, spec in searches if spec is not None]
    with OpenAI(timeout=30, max_retries=2) as ai:
        embeddings = ai.embeddings.create(
            model=model, dimensions=dimensions, encoding_format="float",
            input=[". ".join([spec.clothing_type, spec.style, *spec.details]) for _, spec in searches],
        )
    user_id = str(state["user_id"])
    matches = {}
    for (category, _), embedding in zip(searches, sorted(embeddings.data, key=lambda item: item.index), strict=True):
        result = collection.query(
            query_embeddings=[embedding.embedding], n_results=10,
            where={"$and": [{"user_id": user_id}, {"category": category}]},
            include=["distances"],
        )
        for item_id, distance in zip(result["ids"][0], result["distances"][0], strict=True):
            matches[item_id] = (category, 1 - distance)
    if not matches:
        return {"candidates": []}

    db = _user_db(state)
    rows = (
        db.table("clothes").select(_CLOTHING_COLUMNS)
        .eq("user_id", user_id).in_("id", list(matches)).execute().data
    )
    candidates = [
        {"item": ClothingItem.model_validate(row), "similarity": matches[row["id"]][1]}
        for row in rows if row["category"] == matches[row["id"]][0]
    ]
    # 카테고리별 최고 후보를 확보해 한 종류가 10개를 모두 차지하지 않도록 한다.
    candidates.sort(key=lambda candidate: candidate["similarity"], reverse=True)
    best_by_category = {}
    for candidate in candidates:
        best_by_category.setdefault(candidate["item"].category, candidate["item"].id)
    candidates.sort(
        key=lambda candidate: candidate["item"].id == best_by_category[candidate["item"].category],
        reverse=True,
    )
    return {"candidates": candidates[:10]}


def rank_by_preferences(state: RecommendationState) -> dict:
    candidates = state["candidates"]
    if not candidates:
        return {"ranked_candidates": [], "preference_scores": {}}
    rows = (
        _user_db(state).table("clothing_preferences").select("clothing_id,score")
        .eq("user_id", str(state["user_id"]))
        .in_("clothing_id", [str(candidate["item"].id) for candidate in candidates])
        .execute().data
    )
    scores = {str(row["clothing_id"]): float(row["score"]) for row in rows}
    ranked = sorted(
        candidates,
        key=lambda candidate: (scores.get(str(candidate["item"].id), 0), candidate["similarity"]),
        reverse=True,
    )
    return {"ranked_candidates": ranked, "preference_scores": scores}


def select_outfits(state: RecommendationState) -> dict:
    plan = state["outfit_plan"]
    selected = state.get("selected_clothing")
    fixed = selected["item"] if selected else None
    groups = []
    for category in _CATEGORIES:
        if fixed is not None and fixed.category == category:
            groups.append([{"item": fixed, "similarity": None}])
        elif getattr(plan, category) is not None:
            groups.append([candidate for candidate in state["ranked_candidates"] if candidate["item"].category == category])

    scores = state["preference_scores"]
    combinations = []
    for group in product(*groups):
        items = [candidate["item"] for candidate in group]
        similarities = [candidate["similarity"] for candidate in group if candidate["similarity"] is not None]
        average = sum(similarities) / len(similarities)
        preference = sum(scores.get(str(item.id), 0) for item in items)
        combinations.append((preference, average, items))
    combinations.sort(key=lambda combination: combination[:2], reverse=True)
    picks = combinations[:3] + [combo for combo in combinations[3:] if combo[1] >= _HIGH_SIMILARITY][:3]
    return {"outfits": [Outfit(id=uuid4(), items=items, reason=plan.reason) for _, _, items in picks]}


def save_recommendation(state: RecommendationState) -> dict:
    db = _user_db(state)
    outfits = state["outfits"]
    item_ids = {str(item.id) for outfit in outfits for item in outfit.items}
    photos = {}
    if item_ids:
        rows = (
            db.table("clothes").select("id,image_bucket,image_path")
            .eq("user_id", str(state["user_id"])).in_("id", sorted(item_ids)).execute().data
        )
        for row in rows:
            if row.get("image_bucket") and row.get("image_path"):
                signed = db.storage.from_(row["image_bucket"]).create_signed_url(row["image_path"], 600)
                photos[str(row["id"])] = signed["signedURL"]
    response = RecommendationResponse(
        recommendation_id=uuid4(),
        outfits=[
            outfit.model_copy(update={
                "items": [
                    item.model_copy(update={"photo_url": photos.get(str(item.id))})
                    for item in outfit.items
                ],
            })
            for outfit in outfits
        ],
    )
    db.table("recommendation_history").insert({
        "id": str(response.recommendation_id),
        "user_id": str(state["user_id"]),
        "query": state["query"],
        "outfits": [outfit.model_dump(mode="json", exclude={"items": {"__all__": {"photo_url"}}}) for outfit in response.outfits],
    }).execute()
    return {"response": response}


recommendation_nodes = {
    "load_selected_clothing": load_selected_clothing,
    "fetch_weather": fetch_weather,
    "analyze_intent": analyze_intent,
    "generate_outfit_plan": generate_plan,
    "retrieve_candidates": retrieve_candidates,
    "rank_by_preferences": rank_by_preferences,
    "select_outfits": select_outfits,
    "save_recommendation": save_recommendation,
}
