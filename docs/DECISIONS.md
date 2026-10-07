# 설계 결정 기록 (Supervisor 개선)

형식: "`<항목>`: <대안>으로 하면 <문제>이므로, <선택>을 선정했다." 코드 위치는 괄호로 적는다.

## 패턴·라우팅

- `오케스트레이션 패턴`: 기존 Distributed(단계별 master 게이트 체인)로 하면 실행 순서가 엣지에 묶여 근거가 부족한 관점만 골라 다시 부를 수 없으므로, 단일 Supervisor가 State를 보고 다음 노드를 고르는 Supervisor 패턴을 선정했다. (`supervisor/`, `graph.py`)
- `라우팅 지점`: 게이트 노드마다 conditional edge를 두면 라우팅 규칙이 그래프 여러 곳에 흩어져 순서가 다시 하드코딩되므로, `add_conditional_edges("supervisor", route)` 하나만 두고 결정은 순수 함수 `policy.decide`에 모았다. (`graph.py`, `supervisor/policy.py`)
- `결정과 라우터 분리`: 라우터 함수 안에서 판단하면 결정 사유를 State·로그에 남길 수 없으므로, supervisor 노드가 결정을 `next_agents`·`last_decision`에 쓰고 라우터는 `next_agents`만 읽게 했다. (`supervisor/router.py`)
- `관점 실행 순서`: 기술 조사를 먼저 고정하면 시장·이해관계자·도메인 에이전트가 기술 조사 결과를 입력으로 쓰지 않는데도(데이터 의존성 없음) 대기만 늘어나므로, 미수집 4관점을 State에서 골라 동시에 fan-out하도록 했다. (`supervisor/policy.py::_perspective_step`)
- `병렬 실행 방식`: 정적 병렬 엣지로 하면 항상 같은 묶음만 실행되어 부족한 관점 1개만 다시 보낼 수 없으므로, 실행 대상 수가 State에 따라 바뀌는 `Send` 동적 fan-out을 선정했다. (`supervisor/router.py::route`)
- `재작업 범위`: 부족 시 전 관점을 다시 돌리면 충분한 관점의 LLM·웹 호출이 낭비되고 결과가 흔들리므로, `sufficient=False`인 관점만 `missing`을 재검색 질의(`retry_queries`)로 넘겨 다시 부르게 했다. (`supervisor/policy.py`, `tools/__init__.py`)
- `종합 단계 추가 근거 요청`: 종합이 요청한 관점을 무시하고 공백으로만 남기면 회복 가능한 근거 부족까지 포기하게 되므로, `needs_source_agents`의 관점만 재시도 한도 안에서 다시 부르고 한도를 넘으면 근거 공백으로 기록했다. (`supervisor/policy.py::_synthesis_step`)
- `하위 노드 무효화`: 관점을 다시 조사한 뒤 이전 종합·보고서를 그대로 쓰면 보고서가 새 근거를 반영하지 않으므로, 관점 재할당 시 `node_status`의 synthesis·report·quality_evaluator를 pending으로 되돌렸다. (`supervisor/policy.py::invalidate_downstream`)

## State Schema (DEV_PLAN §5 7항목)

- `제어 vs 페이로드 분리`: 한 딕셔너리에 섞으면 라우팅 조건이 결과 본문 구조에 의존해 프롬프트를 바꿀 때 라우팅이 깨지므로, Supervisor는 제어 필드(`perspective_status, node_status, retry_counts, eval_result, step_count`)와 결과의 `sufficient/missing`만 읽도록 분리했다. (`state.py::GraphState`)
- `관측성 위치`: 로그를 State의 `operator.add` 리스트에 쌓으면 체크포인트마다 전체 이력이 복제되어 커지므로, 결정 로그 본문은 `outputs/decisions_{trace_id}.jsonl`과 LangSmith에 두고 State에는 직전 결정 1건(`last_decision`)만 남겼다. (`observability.py`)
- `지속성 비용`: 원문 RAG 캐시를 State에 넣으면 `final_state.json`이 848KB까지 커지고 체크포인트마다 저장되므로, 캐시는 `data/cache/{trace_id}/rag_{agent}.json` 디스크에 두고 State에는 `cache_keys`만, `evidence`는 발췌 1,600자 상한과 dedup reducer로 병합했다. (`rag/cache.py`, `state.py::merge_evidence`)
- `상관 키`: 키를 따로 쓰면 트레이스·State·로그를 사람이 손으로 맞춰야 하므로, uuid4 `trace_id` 하나를 LangGraph `thread_id`·LangSmith metadata·결정 로그 파일명에 함께 썼다. (`observability.py::run_config`)
- `재개/복구`: 메모리 체크포인터는 프로세스가 죽으면 사라져 15분짜리 실행을 처음부터 다시 해야 하므로, `SqliteSaver`(thread_id=trace_id)와 `node_status`·`last_error`·`retry_counts`로 마지막 체크포인트부터 `--resume`하게 했다. (`app.py::open_checkpointer`)
- `동시 처리`: reducer 없이 `Send`로 병렬 실행하면 같은 키에 동시에 쓸 때 `InvalidUpdateError`가 나거나 마지막 값만 남으므로, `perspectives·node_status·retry_counts·last_error·feedback`은 키 단위 dict merge, `evidence`는 dedup-append, `gaps`는 순서 유지 중복 제거 reducer로 정했다. (`state.py`)
- `종료 보장`: `recursion_limit`만 두면 상한에 걸릴 때 예외로 죽어 보고서가 남지 않으므로, Supervisor가 `step_count > MAX_STEPS`를 먼저 감지해 조사를 멈추고 근거 공백을 명시한 뒤 종합→보고서→평가까지 마치고 정상 종료하게 했다(재시도 상한·`recursion_limit`은 2·3차 그물). (`supervisor/policy.py::decide`, `config.py`)

## 실패 처리·품질 평가

- `평가 기반 재조사 질의`: 편향·커버리지 미달 사유 문장을 그대로 재검색 질의로 넘기면 검색 결과가 나오지 않으므로, 사유는 `missing`으로 전달하고 질의는 반대 방향(우려/긍정)·독립 출처를 찾는 힌트로 바꿨다. (`supervisor/policy.py::_eval_feedback`)
- `에이전트 예외 처리`: 예외를 그대로 올리면 그래프 전체가 죽어 보고서가 남지 않으므로, 래퍼가 예외를 `node_status=failed`·`last_error`로 바꾸고 Supervisor가 한도 안에서 재시도, 넘으면 제외 후 근거 공백으로 기록하게 했다. (`supervisor/guard.py`)
- `품질 평가 방식`: 형식 검사만 하면 근거가 주장을 뒷받침하는지 볼 수 없고 LLM Judge만 쓰면 실행마다 판정이 바뀌므로, 규칙 검사를 하드 게이트·LLM Judge(`EvalVerdict`)를 내용 게이트로 쓰는 Hybrid를 선정하고 규칙 실패는 Judge가 뒤집지 못하게 했다. (`evaluation/quality.py::combine`)
- `미달 원인별 경로`: 미달이면 항상 보고서만 다시 쓰면 근거 자체가 편향·누락된 경우 같은 결함이 반복되므로, 편향 통제·관점 커버리지 미달은 원인 관점 재조사, Groundedness·중립성 미달은 보고서 재작성으로 나눴다. (`supervisor/policy.py::_report_and_eval_step`)
- `평가 노드 위치`: 보고서 후 Supervisor를 한 번 더 거쳐 평가로 가면 평가 없이 종료하는 경로가 생길 수 있으므로, 평가 노드를 Supervisor 측 품질 게이트로 두고 `report → quality_evaluator → supervisor` 고정 엣지를 선정했다(하위 에이전트끼리의 엣지는 0개). (`graph.py`)
- `근거 공백 인정`: 재조사 한도를 넘긴 편향·커버리지 미달을 끝까지 실패로 두면 공개 근거가 실제로 없는 경우에도 보고서가 영원히 미검증이 되므로, Supervisor가 `gaps`에 `관점/기술:` 형식으로 기록하고 보고서 7장에 코드로 명시한 경우에만 규칙상 인정했다. (`evaluation/quality.py::_acknowledged`, `agents/report.py::gap_section`)
- `편향 규칙의 방향성`: 도메인 `조건부`를 우려로 세면 조건부 적합 판정만 있는 기술이 한쪽 근거뿐이라고 오판되므로, `조건부`는 `혼재`와 같이 긍정·우려 양쪽을 담은 판정으로 보았다. (`evaluation/quality.py`)
- `중립성 규칙`: 금지어만 찾으면 "순위나 추천을 제시하지 않는다" 같은 면책 문장이 걸려 재작성이 낭비되므로, 같은 문장에 부정·면책 표현이 있으면 제외했다(이전 실행 보고서로 오탐 0건 확인). (`evaluation/quality.py::check_neutrality`)
- `수치 문장 인용 규칙`: 모든 숫자를 검사하면 TRL 등급·날짜·코드가 만든 증거 균형표까지 걸리므로, 단위가 붙은 수치(%, 배, GB, ms, 토큰 등)가 있는 문장·표 행만 인용을 요구하고 설계 표·산식([D])은 제외했다. (`evaluation/quality.py::check_groundedness`)
- `종료 상태`: 성공/실패 두 값으로 하면 "평가는 통과했지만 근거 공백이 있다"와 "평가 미통과"가 구분되지 않으므로, `completed`·`completed_with_gaps`·`unverified` 세 값으로 나눴다. (`supervisor/policy.py::_Builder.end`)

## 제출물·운영

- `보고서 10p 검사`: 글자 수로 페이지를 추정하면 표·소제목 비중에 따라 오차가 크므로, 내보내기와 같은 reportlab 조판으로 메모리 렌더링해 실제 페이지 수를 셌다. (`reporting/export.py::report_page_count`)
- `LangSmith 연결`: 그래프 노드만 트레이스하면 노드 안의 LLM 호출 비용·지연을 볼 수 없으므로, 환경변수 트레이싱에 더해 OpenAI 클라이언트를 `wrap_openai`로 감싸 같은 run 아래에 남겼다. (`llm.py::_traced`)
- `그래프 이미지`: `draw_mermaid_png`만 쓰면 mermaid.ink에 접근할 수 없는 환경에서 이미지가 생성되지 않으므로, 실패 시 컴파일 그래프의 실제 노드·엣지를 Pillow로 그리고 Mermaid 원문도 함께 저장했다. (`scripts/export_graph.py`)
- `--report-only`: 별도 루프 코드로 보고서만 다시 돌리면 그래프와 재시도 규칙이 갈라지므로, 같은 Supervisor 그래프에 관점 재조사 한도 0인 Policy를 넣어 재사용했다. (`app.py::report_only`)
- `오프라인 테스트`: 실제 API로 흐름을 검증하면 키가 필요하고 실행마다 결과가 달라 재현이 안 되므로, 스키마 종류로 에이전트를 구분해 실행 회차별로 부족·편향·예외·중단을 주입하는 Fake LLM·Web·RAG를 만들었다. (`tests/fakes.py`)
