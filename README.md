# P-Project
수정될 가능성 다분

추천 시스템/
├── database/
│   ├── init.py
│   ├── clients.py          # Supabase·ChromaDB 연결
│   ├── clothes.py          # 의류 상태·사진·보관 위치 조회
│   ├── vectors.py          # 의류 벡터 저장·검색
│   ├── preferences.py      # 사용자 선호도 조회·저장
│   ├── history.py          # 추천 이력·피드백 저장
│   ├── index_clothes.py    # 보유 의류 임베딩 등록·갱신
│   └── schema.sql          # DB 테이블·접근 정책
│
├── recommendation/
│   ├── init.py
│   ├── api.py              # 추천·피드백 요청 접수
│   ├── graph.py            # LangGraph 노드·연결·조건 분기
│   ├── state.py            # 노드 사이에 전달할 공통 상태
│   ├── nodes.py            # 각 단계의 실행 함수
│   ├── schemas.py          # 입력·LLM 출력·최종 응답 형식
│   ├── prompts.py          # 의도 분석·코디 생성 프롬프트
│   ├── llm.py              # LLM 호출
│   ├── embedding.py        # 등록·검색 공통 임베딩 생성
│   ├── weather.py          # 날씨 조회
│   ├── retrieval.py        # 적격 의류 필터링·벡터 검색
│   ├── ranking.py          # 코디 조합·유사도·선호도 정렬
│   ├── feedback.py         # 별점·선택에 따른 선호도 갱신
│   ├── speech.py           # 음성 인식
│   ├── led.py              # 보관 위치 점등
│   ├── config.py           # 모델·환경변수·추천 기준
│   └── ui.py               # 화면이 별도로 없다면 추가
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md