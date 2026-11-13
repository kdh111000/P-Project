"""코디 추천의 분기와 실행 순서를 정의한다.

state.py와 nodes.py가 구현되면 build_graph에 상태 타입과 노드를 전달한다.
각 노드는 상태를 받고 갱신할 필드의 딕셔너리를 반환한다.
"""
'''
# ┌──────────────────────────────────┐
# │ START · 코디 추천 요청           │
# └────────────────┬─────────────────┘
#                  ▼
# ┌──────────────────────────────────┐
# │ 선택한 옷 ID가 있는가?           │
# └────────┬──────────────────┬──────┘
#          │ 없음             │ 있음
#          │                  ▼
#          │   ┌────────────────────────────────┐
#          │   │ load_selected_clothing         │
#          │   │ 사용자 소유 확인 및 옷 정보 조회 │
#          │   └───────────────┬────────────────┘
#          │                   │
#          └───────┬───────────┘
#                  ▼
# ┌──────────────────────────────────┐
# │ fetch_weather                    │
# │ 현재 날씨 조회 · 지역 생략 시 서울 │
# └────────────────┬─────────────────┘
#                  ▼
# ┌──────────────────────────────────┐
# │ analyze_intent                   │
# │ LLM으로 요청·날씨·선택 의류 분석  │
# └────────────────┬─────────────────┘
#                  ▼
# ┌──────────────────────────────────┐
# │ generate_outfit_plan             │
# │ LLM으로 상의·하의·외투 조건 생성  │
# └────────────────┬─────────────────┘
#                  ▼
# ┌──────────────────────────────────┐
# │ retrieve_candidates              │
# │ ChromaDB에서 보유 의류 검색       │
# │ 종류별 후보 확보 · 전체 최대 10개 │
# └────────────────┬─────────────────┘
#                  ▼
# ┌──────────────────────────────────┐
# │ rank_by_preferences              │
# │ Supabase 선호도순 · 동점은 유사도순│
# └────────────────┬─────────────────┘
#                  ▼
# ┌──────────────────────────────────┐
# │ select_outfits                   │
# │ 상위 3개 + 평균 유사도 0.85 이상  │
# │ 최대 3개 추가 · 선택한 옷은 고정  │
# └────────────────┬─────────────────┘
#                  ▼
# ┌──────────────────────────────────┐
# │ save_recommendation              │
# │ 사진 URL 생성 및 추천 이력 저장   │
# └────────────────┬─────────────────┘
#                  ▼
# ┌──────────────────────────────────┐
# │ END · 추천 ID와 코디 목록 반환   │
# └──────────────────────────────────┘
'''
from collections.abc import Callable, Mapping
from typing import Literal

from langgraph.graph import END, START, StateGraph

from .errors import RecommendationUnavailable

_STEPS = (
    "fetch_weather",
    "analyze_intent",
    "generate_outfit_plan",
    "retrieve_candidates",
    "rank_by_preferences",
    "select_outfits",
    "save_recommendation",
)


def _route_request(state: Mapping[str, object]) -> Literal["selected_clothing", "text_only"]:
    if state.get("selected_clothing_id") is not None:
        return "selected_clothing"
    return "text_only"


def build_graph(
    state_schema: type,
    nodes: Mapping[str, Callable],
):
    """추천 실행에 사용할 컴파일된 LangGraph를 반환한다.

    입력 상태에는 query, user_id, location과 선택적인 selected_clothing_id가 있다.
    load_selected_clothing은 사용자 소유의 옷 한 벌을 조회해 selected_clothing에
    기록한다. 사용자 소유의 옷을 찾지 못하면 오류를 발생시킨다.
    analyze_intent와 generate_outfit_plan은 query, weather 및 selected_clothing이
    있으면 해당 옷의 속성도 LLM 입력에 포함한다. 선택한 옷은 코디 조합에 고정한다.

    retrieve_candidates는 사용자 보유 의류에서 필요한 종류별 최고 후보를 먼저
    확보하고 유사도 순으로 최대 10개 후보를 채운다.
    rank_by_preferences는 사용자 선호도로 정렬하고,
    select_outfits는 상위 3개와 유사도가 매우 높은 후보를 최대 3개 더 선택한다.
    save_recommendation은 표시용 결과를 만들고 반환 전에 추천 이력을 저장한다.
    """
    graph = StateGraph(state_schema)
    graph.add_node("load_selected_clothing", nodes["load_selected_clothing"])
    for step in _STEPS:
        graph.add_node(step, nodes[step])

    graph.add_conditional_edges(
        START,
        _route_request,
        {"selected_clothing": "load_selected_clothing", "text_only": _STEPS[0]},
    )
    graph.add_edge("load_selected_clothing", _STEPS[0])
    for source, target in zip(_STEPS, _STEPS[1:]):
        graph.add_edge(source, target)
    graph.add_edge(_STEPS[-1], END)
    return graph.compile()


def recommend(user, payload):
    """사용자와 요청을 받아 추천 그래프를 실행한다."""
    try:
        from .nodes import recommendation_nodes
        from .state import RecommendationState
    except ModuleNotFoundError as exc:
        if exc.name not in {f"{__package__}.nodes", f"{__package__}.state"}:
            raise
        raise RecommendationUnavailable("추천 그래프에 상태와 노드가 아직 연결되지 않았습니다.") from None

    graph = build_graph(RecommendationState, recommendation_nodes)
    result = graph.invoke({
        "user_id": user.id,
        "access_token": user.access_token,
        **payload.model_dump(),
    })
    return result["response"]
