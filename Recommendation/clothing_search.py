"""코디 조건별로 ChromaDB를 검색하고 Supabase 사진 정보를 결합한다."""

import logging
import math
import os
from pathlib import Path
from typing import Literal
from uuid import UUID

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from DataBase.Scripts.clients import get_chroma_client, get_supabase_client
from .schemas import OutfitPlan


class ClothingMatch(BaseModel):
    id: UUID
    name: str
    category: Literal["top", "bottom", "outer"]
    similarity: float = Field(ge=-1, le=1)
    photo_url: str | None
    image_bucket: str | None
    image_path: str | None


class ClothingSearchResult(BaseModel):
    top: list[ClothingMatch]
    bottom: list[ClothingMatch]
    outer: list[ClothingMatch]


class SelectedClothingImage(BaseModel):
    id: UUID
    name: str
    category: Literal["top", "bottom", "outer"]
    photo_url: str | None = None


class OutfitWithImages(BaseModel):
    plan: OutfitPlan
    matches: ClothingSearchResult
    selected_clothing: SelectedClothingImage | None = None


def search_clothes(
    plan: OutfitPlan,
    *,
    user_id: str | UUID | None = None,
    top_k: int = 3,
    client: OpenAI | None = None,
    db=None,
    collection=None,
    excluded_category: Literal["top", "bottom", "outer"] | None = None,
) -> ClothingSearchResult:
    """각 종류별 상위 top_k개 반환. similarity는 1 - cosine distance다.

    db를 생략하면 기존 서버용 Supabase 키를 사용한다. 사용자 세션의 RLS가
    필요하면 해당 세션으로 인증한 db 클라이언트를 전달한다.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
        raise ValueError("top_k는 1~10 사이 정수여야 합니다.")
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
    owner = str(UUID(str(user_id or os.getenv("IMPORT_USER_ID", ""))))
    db = db if db is not None else get_supabase_client()
    if collection is None:
        collection = get_chroma_client().get_collection(
            os.getenv("CHROMA_COLLECTION", "clothes_descriptions_v1"),
            embedding_function=None,
        )
    metadata = collection.metadata or {}
    model, dimensions = metadata.get("embedding_model"), metadata.get("embedding_dimensions")
    if not isinstance(model, str) or not isinstance(dimensions, int) or metadata.get("hnsw:space") != "cosine":
        raise ValueError("Chroma 컬렉션의 모델·차원·cosine 설정을 확인해 주세요.")
    specs = [(category, getattr(plan, category)) for category in ("top", "bottom", "outer")]
    specs = [(category, spec) for category, spec in specs if spec is not None and category != excluded_category]
    texts = [". ".join([spec.clothing_type, spec.style, *spec.details]) for _, spec in specs]
    owned_client = client is None
    ai = client if client is not None else OpenAI(timeout=30, max_retries=2)
    try:
        embeddings = ai.embeddings.create(
            model=model, dimensions=dimensions, input=texts, encoding_format="float"
        )
    finally:
        if owned_client:
            ai.close()
    matches = {"top": [], "bottom": [], "outer": []}
    for (category, _), embedding in zip(specs, sorted(embeddings.data, key=lambda item: item.index), strict=True):
        if len(embedding.embedding) != dimensions:
            raise ValueError("검색 임베딩의 차원이 등록된 의류와 다릅니다.")
        result = collection.query(
            query_embeddings=[embedding.embedding], n_results=top_k,
            where={"$and": [{"user_id": owner}, {"category": category}]},
            include=["distances"],
        )
        ids, distances = result["ids"][0], result["distances"][0]
        if not ids:
            continue
        rows = (
            db.table("clothes").select("id,name,category,image_bucket,image_path")
            .eq("user_id", owner).eq("category", category).in_("id", ids).execute().data
        )
        by_id = {str(row["id"]): row for row in rows}
        for clothing_id, distance in zip(ids, distances, strict=True):
            row = by_id.get(clothing_id)
            if row is None or distance is None or not math.isfinite(distance):
                continue
            photo_url = None
            if row.get("image_bucket") and row.get("image_path"):
                try:
                    signed = db.storage.from_(row["image_bucket"]).create_signed_url(row["image_path"], 600)
                    photo_url = signed["signedURL"]
                except Exception:
                    logging.warning("의류 사진 URL 생성 실패: %s", clothing_id)
            matches[category].append(ClothingMatch(
                id=row["id"], name=row["name"], category=category,
                similarity=max(-1, min(1, 1 - distance)), photo_url=photo_url,
                image_bucket=row.get("image_bucket"), image_path=row.get("image_path"),
            ))
    return ClothingSearchResult(**matches)
