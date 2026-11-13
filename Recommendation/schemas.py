"""추천 API 입력, LLM 입출력, 최종 응답 모델."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .weather import WeatherInfo

UserQuery = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: UserQuery
    location: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)] | None = None
    selected_clothing_id: UUID | None = None


class ClothingItem(BaseModel):
    id: UUID
    name: str
    category: Literal["top", "bottom", "outer"]
    photo_url: str | None = None
    storage_location: str | None = None


class Outfit(BaseModel):
    id: UUID
    items: list[ClothingItem]
    reason: str


class RecommendationResponse(BaseModel):
    recommendation_id: UUID
    outfits: list[Outfit]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outfit_id: UUID
    rating: Annotated[int, Field(strict=True, ge=1, le=5)] | None = None
    selected: bool = False

    @property
    def has_feedback(self) -> bool:
        return self.rating is not None or self.selected


class GarmentSpec(BaseModel):
    """한 카테고리에서 검색할 옷의 조건."""

    model_config = ConfigDict(extra="forbid")

    clothing_type: str
    style: str
    details: list[str]


class OutfitPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top: GarmentSpec
    bottom: GarmentSpec
    outer: GarmentSpec | None
    reason: str


class OutfitPlanInput(BaseModel):
    query: UserQuery
    weather: WeatherInfo
    intent: str | None = None
    selected_clothing: dict[str, str] | None = None
