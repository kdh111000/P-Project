"""추천 그래프의 입력과 노드별 결과를 정의한다."""

from uuid import UUID

from pydantic import SecretStr
from typing_extensions import Required, TypedDict

from .api import ClothingItem, Outfit, RecommendationResponse
from .outfit_plan import OutfitPlan
from .weather import WeatherInfo


class SelectedClothing(TypedDict, total=False):
    """사용자가 고른 의류의 기본 정보와 LLM에 전달할 설명."""

    item: Required[ClothingItem]
    description: str | None


class ClothingCandidate(TypedDict):
    """의류 검색 결과 한 건. 유사도가 높을수록 코디 조건에 가깝다."""

    item: ClothingItem
    similarity: float


class RecommendationState(TypedDict, total=False):
    """한 추천 요청의 상태. 각 노드는 자신이 생성한 필드만 반환한다."""

    user_id: Required[UUID]
    query: Required[str]
    access_token: SecretStr  # 사용자 범위의 DB 조회에만 사용한다.
    location: str | None
    selected_clothing_id: UUID | None

    selected_clothing: SelectedClothing
    weather: WeatherInfo
    intent: str
    outfit_plan: OutfitPlan

    candidates: list[ClothingCandidate]  # 보유 의류에서 검색한 상위 후보
    ranked_candidates: list[ClothingCandidate]  # 사용자 선호도 순
    outfits: list[Outfit]
    response: RecommendationResponse  # 이력 저장 후 API로 반환할 결과
