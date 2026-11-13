"""추천 실행 중 호출자가 처리할 수 있는 오류."""


class RecommendationError(RuntimeError):
    """추천을 완료할 수 없다."""


class RecommendationAuthRequired(RecommendationError):
    """사용자 인증 정보가 필요하다."""


class SelectedClothingNotFound(RecommendationError):
    """사용자 옷장에서 선택한 옷을 찾을 수 없다."""


class InvalidRecommendationLocation(RecommendationError):
    """날씨를 조회할 지역 입력을 처리할 수 없다."""


class RecommendationUnavailable(RecommendationError):
    """추천에 필요한 서비스나 설정을 사용할 수 없다."""


class IntentAnalysisFailed(RecommendationError):
    """LLM에서 완성된 의도 분석을 받지 못했다."""
