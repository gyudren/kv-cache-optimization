# 설계 결정 기록 (Supervisor 개선)

형식: "`<항목>`: <대안>으로 하면 <문제>이므로, <선택>을 선정했다." 코드 위치는 괄호로 적는다.

## 패턴·라우팅

- `오케스트레이션 패턴`: 기존 Distributed(단계별 master 게이트 체인)로 하면 실행 순서가 엣지에 묶여 근거가 부족한 관점만 골라 다시 부를 수 없으므로, 단일 Supervisor가 State를 보고 다음 노드를 고르는 Supervisor 패턴을 선정했다. (`supervisor/`, `graph.py`)
- `Supervisor 판단 방식(규칙 vs LLM)`: Supervisor를 LLM 라우터로 하면 같은 State에서도 실행마다 다음 노드가 바뀌어 재현·테스트가 불가능하고 매 진입마다 LLM 호출 비용과 지연이 붙으므로, 라우팅 입력이 모두 구조화된 제어 필드(충분도·노드 상태·재시도 횟수·평가 결과)라는 점을 이용해 결정적 규칙 함수 `policy.decide`를 선정했다(내용 판단은 각 에이전트와 LLM Judge가 맡는다). (`supervisor/policy.py`, `docs/SUPERVISOR_POLICY.md`)
- `라우팅 지점`: 게이트 노드마다 conditional edge를 두면 라우팅 규칙이 그래프 여러 곳에 흩어져 순서가 다시 하드코딩되므로, `add_conditional_edges("supervisor", route)` 하나만 두고 결정은 순수 함수 `policy.decide`에 모았다. (`graph.py`, `supervisor/policy.py`)
- `결정과 라우터 분리`: 라우터 함수 안에서 판단하면 결정 사유를 State·로그에 남길 수 없으므로, supervisor 노드가 결정을 `next_agents`·`last_decision`에 쓰고 라우터는 `next_agents`만 읽게 했다. (`supervisor/router.py`)
- `관점 실행 순서`: 기술 조사를 먼저 고정하면 시장·이해관계자·도메인 에이전트가 기술 조사 결과를 입력으로 쓰지 않는데도(데이터 의존성 없음) 대기만 늘어나므로, 미수집 4관점을 State에서 골라 동시에 fan-out하도록 했다. (`supervisor/policy.py::_perspective_step`)
- `병렬 실행 방식`: 정적 병렬 엣지로 하면 항상 같은 묶음만 실행되어 부족한 관점 1개만 다시 보낼 수 없으므로, 실행 대상 수가 State에 따라 바뀌는 `Send` 동적 fan-out을 선정했다. (`supervisor/router.py::route`)
- `재작업 범위`: 부족 시 전 관점을 다시 돌리면 충분한 관점의 LLM·웹 호출이 낭비되고 결과가 흔들리므로, `sufficient=False`인 관점만 `missing`을 재검색 질의(`retry_queries`)로 넘겨 다시 부르게 했다. (`supervisor/policy.py`, `tools/__init__.py`)
- `충분성 판단 주체`: 에이전트의 자기 보고(`sufficient`)만 믿으면 출처 1개로도 "충분"이 통과해 편향이 보고서 단계에서야 드러나므로, Supervisor가 관점·기술별 고유 출처 수(≥2)를 결정적으로 다시 검사해 부족하면 그 관점만 재조사시킨다. (`supervisor/policy.py::classify`, `evaluation/quality.py::evidence_shortfalls`)
- `재작업 지시 전달`: 이전에 충분했던 RAG 답을 그대로 재사용하면 재작업 지시가 있어도 같은 근거만 돌아오므로, 지시(missing·재검색 힌트)의 지문을 캐시 키에 넣어 지시가 겨냥한 질문(기술명·D-코드 기준)만 다시 검색하고 지시를 RAG 질의 계획과 에이전트 프롬프트(`SUPERVISOR REWORK REQUEST`)에 넣었다. (`rag/workflow.py::answer_with_cache`, `tools/__init__.py::rework_note`)
- `Send 페이로드`: `Send(name, state)`로 State 전체를 넘기면 체크포인트의 대기 작업마다 evidence·보고서가 fan-out 수만큼 복제되므로, 관점 에이전트가 읽는 필드(`trace_id·retry_counts·feedback·user_query·step_count`)만 넘겼다. (`supervisor/router.py::SEND_KEYS`)
- `스레드 컨텍스트`: 에이전트 안의 질문 병렬 처리를 일반 `ThreadPoolExecutor`로 하면 contextvars가 끊겨 LLM·웹 호출이 LangSmith에서 그래프 run 밖의 고아 run이 되므로, `ContextThreadPoolExecutor`로 실행 설정·부모 run을 워커 스레드에 복사했다(테스트로 trace_id 전파 확인). (`agents/technology.py`, `agents/domain.py`)
- `종합 단계 추가 근거 요청`: 종합이 요청한 관점을 무시하고 공백으로만 남기면 회복 가능한 근거 부족까지 포기하게 되므로, `needs_source_agents`의 관점만 재시도 한도 안에서 다시 부르고 한도를 넘으면 근거 공백으로 기록했다. (`supervisor/policy.py::_synthesis_step`)
- `하위 노드 무효화`: 관점을 다시 조사한 뒤 이전 종합·보고서를 그대로 쓰면 보고서가 새 근거를 반영하지 않으므로, 관점 재할당 시 `node_status`의 synthesis·report·quality_evaluator를 pending으로 되돌렸다. (`supervisor/policy.py::invalidate_downstream`)

## State Schema (DEV_PLAN §5 7항목)

- `제어 vs 페이로드 분리`: 한 딕셔너리에 섞으면 라우팅 조건이 결과 본문 구조에 의존해 프롬프트를 바꿀 때 라우팅이 깨지므로, Supervisor는 제어 필드(`perspective_status, node_status, retry_counts, eval_result, step_count`)와 결과의 `sufficient/missing`만 읽도록 분리했다. (`state.py::GraphState`)
- `관측성 위치`: 로그를 State의 `operator.add` 리스트에 쌓으면 체크포인트마다 전체 이력이 복제되어 커지므로, 결정 로그 본문은 `outputs/decisions_{trace_id}.jsonl`과 LangSmith에 두고 State에는 직전 결정 1건(`last_decision`)만 남겼다. (`observability.py`)
- `지속성 비용`: 원문 RAG 캐시와 Evidence 발췌 원문을 State에 넣으면 이전 실행 기준 `final_state.json` 848KB(evidence만 491KB)가 체크포인트마다 다시 저장되므로, RAG 캐시는 `data/cache/{trace_id}/rag_{agent}.json`, 발췌 원문은 `evidence_{agent}.json` 디스크 저장소에 두고 State에는 300자 축약본·`excerpt_ref`·`cache_keys`만 남겼다(실측: 이전 실행 evidence 491,320B → 227,639B, Fake 재작업 실행 체크포인트 누적 3,968,002B → 1,856,254B). 보고서·평가는 `hydrate()`로 원문을 쓴다. (`evidence_store.py`, `rag/cache.py`, `scripts/measure_state_size.py`)
- `상관 키`: 키를 따로 쓰면 트레이스·State·로그를 사람이 손으로 맞춰야 하므로, uuid4 `trace_id` 하나를 LangGraph `thread_id`·LangSmith metadata·결정 로그 파일명에 함께 썼다. (`observability.py::run_config`)
- `재개/복구`: 메모리 체크포인터(`MemorySaver`)는 프로세스가 죽으면 사라져 15분짜리 실행을 처음부터 다시 해야 하고, Postgres 체크포인터는 단일 사용자 CLI에 DB 서버 운영을 요구하므로, 파일 하나로 끝나는 `SqliteSaver`(thread_id=trace_id)와 `node_status`·`last_error`·`retry_counts`로 마지막 체크포인트부터 `--resume`하게 했다. (`app.py::open_checkpointer`)
- `동시 처리`: reducer 없이 `Send`로 병렬 실행하면 같은 키에 동시에 쓸 때 `InvalidUpdateError`가 나거나 마지막 값만 남으므로, `perspectives·node_status·retry_counts·last_error·feedback`은 키 단위 dict merge, `evidence`는 dedup-append, `gaps`는 순서 유지 중복 제거 reducer로 정했다. (`state.py`)
- `종료 보장`: `recursion_limit`만 두면 상한에 걸릴 때 예외로 죽어 보고서가 남지 않으므로, Supervisor가 `step_count > MAX_STEPS`를 먼저 감지해 조사를 멈추고 근거 공백을 명시한 뒤 종합→보고서→평가까지 마치고 정상 종료하게 했다(재시도 상한·`recursion_limit`은 2·3차 그물). (`supervisor/policy.py::decide`, `config.py`)
- `재시도 상한 값`: 관점 재시도를 0~1회로 하면 질의 재작성 한 번으로 회복되는 일시적 근거 부족도 공백으로 끝나고 3회 이상이면 같은 공개 자료를 반복 검색해 비용(관점당 LLM 호출 수십 회)만 늘므로, 관점 2회·보고서 재작성 2회·종합 1회(표현 보완만)·평가 실행 실패 1회로 정했다(이전 실제 실행 로그에서도 tech 재작성 2회·관점 재할당 2회가 발생해 2회가 실사용 범위). (`config.py::RETRY_LIMITS`)
- `MAX_STEPS = 20`: 10 이하로 하면 Fake 고장 주입 시나리오의 최대 관측치(Judge 4항목 상시 미달 17회, 판정 한쪽 고정 15회)조차 마치기 전에 마무리 모드로 끊기고, 이론적 최악(관점 재조사가 평가 단계에서 하나씩 소진되어 30회 이상)까지 허용하면 실제 실행(진입당 수 분)이 1시간을 넘으므로, 정상 경로(5회)와 관측 최악(17회)은 끝까지 가고 그 이상은 근거 공백을 명시하는 마무리 모드로 끊도록 20으로 정했다. (`config.py`)
- `FINALIZE_STEPS = 5`: 상한 도달 후 마무리는 종합→보고서→평가→종료 4회인데 4로 딱 맞추면 재개 직후 재진입 한 번에도 보고서 없이 하드 종료되므로, 여유 1을 더해 5로 정했다(마무리 중 실패는 재시도하지 않아 4회를 넘지 않는다). (`config.py`)

## 실패 처리·품질 평가

- `평가 기반 재조사 질의`: 편향·커버리지 미달 사유 문장을 그대로 재검색 질의로 넘기면 검색 결과가 나오지 않으므로, 사유는 `missing`으로 전달하고 질의는 반대 방향(우려/긍정)·독립 출처를 찾는 힌트로 바꿨다(규칙 사유가 없는 Judge 단독 미달도 일반 힌트만 사용). (`supervisor/policy.py::_eval_feedback`)
- `에이전트 예외 처리`: 예외를 그대로 올리면 그래프 전체가 죽어 보고서가 남지 않으므로, 래퍼가 예외를 `node_status=failed`·`last_error`로 바꾸고 Supervisor가 한도 안에서 재시도, 넘으면 제외 후 근거 공백으로 기록하게 했다. (`supervisor/guard.py`)
- `품질 평가 방식`: 형식 검사만 하면 근거가 주장을 뒷받침하는지 볼 수 없고 LLM Judge만 쓰면 실행마다 판정이 바뀌므로, 규칙 검사를 하드 게이트·LLM Judge(`EvalVerdict`)를 내용 게이트로 쓰는 Hybrid를 선정하고 규칙 실패는 Judge가 뒤집지 못하게 했다. (`evaluation/quality.py::combine`)
- `미달 원인별 경로`: 미달이면 항상 보고서만 다시 쓰면 근거 자체가 편향·누락된 경우 같은 결함이 반복되므로, 편향 통제·관점 커버리지 미달은 원인 관점 재조사, Groundedness·중립성 미달은 보고서 재작성으로 나눴다. (`supervisor/policy.py::_report_and_eval_step`)
- `평가 노드 위치`: `report → quality_evaluator` 고정 엣지로 하면 보고서 에이전트가 Supervisor를 거치지 않고 다른 노드로 넘기게 되고 실패한 보고서까지 평가(빈 보고서에 Judge 호출)되므로, 보고서를 포함한 모든 작업 노드가 Supervisor로만 돌아오고 Supervisor가 "보고서 정상 완료 + 평가 미실행"일 때만 `evaluate`로 평가 노드를 부르게 했다(마지막 보고서 이후 평가 없이 `end:passed`가 불가능함을 테스트로 강제). (`graph.py`, `supervisor/policy.py::_report_and_eval_step`)
- `근거 공백 인정`: 재조사 한도를 넘긴 편향·커버리지 미달을 끝까지 실패로 두면 공개 근거가 실제로 없는 경우에도 보고서가 영원히 미검증이 되므로, Supervisor가 `관점/기술:` 공백으로 기록한 기술(또는 관점을 실제로 제외한 경우의 `관점:` 공백)에 한해 "결과·근거 없음"만 면제하고, 절 구조(판정 표·서술·기술명)는 공백이 있어도 항상 요구했다. (`evaluation/quality.py::_acknowledged`, `agents/report.py::gap_section`)
- `편향 임계값`: 고유 출처 하한을 1로 하면 단일 기사·단일 페이지로 판정이 확정되고 3 이상이면 공개 자료가 적은 ITME 같은 신기술에서 거의 항상 공백이 되므로 `MIN_DISTINCT_SOURCES=2`, 단일 발행처 비중 상한은 50%면 출처 2개 중 같은 발행처 1개(50%)는 통과하면서 3개 중 2개(67%)부터 걸리는 경계가 모호하고 70% 이상이면 한 매체 편중을 놓치므로 `MAX_SINGLE_SOURCE_SHARE=0.6`(3개 중 2개 이상 같은 발행처면 재조사)으로 정했다. (`config.py`)
- `편향 규칙의 방향성`: 도메인 `조건부`를 우려로 세면 조건부 적합 판정만 있는 기술이 한쪽 근거뿐이라고 오판되므로, `조건부`는 `혼재`와 같이 긍정·우려 양쪽을 담은 판정으로 보았다. (`evaluation/quality.py`)
- `중립성 규칙`: 금지어만 찾으면 "순위나 추천을 제시하지 않는다" 같은 면책 문장이 걸려 재작성이 낭비되므로, 같은 문장에 부정·면책 표현이 있으면 제외했다(이전 실행 보고서로 오탐 0건 확인). (`evaluation/quality.py::check_neutrality`)
- `수치 문장 인용 규칙`: 모든 숫자를 검사하면 TRL 등급·날짜·코드가 만든 증거 균형표까지 걸리므로, 단위가 붙은 수치(%, 배, GB, ms, 토큰 등)가 있는 문장·표 행만 인용을 요구하고 설계 표·산식([D])은 제외했다. (`evaluation/quality.py::check_groundedness`)
- `종료 상태`: 성공/실패 두 값으로 하면 "평가는 통과했지만 근거 공백이 있다"와 "평가 미통과"가 구분되지 않으므로, `completed`·`completed_with_gaps`·`unverified` 세 값으로 나눴다. (`supervisor/policy.py::_Builder.end`)

## 제출물·운영

- `보고서 10p 검사`: 글자 수로 페이지를 추정하면 표·소제목 비중에 따라 오차가 크므로, 내보내기와 같은 reportlab 조판으로 메모리 렌더링해 실제 페이지 수를 셌다. (`reporting/export.py::report_page_count`)
- `조판 미검사 처리`: 한글 폰트가 없을 때 페이지 검사를 조용히 건너뛰면 10p 초과 보고서가 "검증 통과"로 제출될 수 있으므로, 검사 불가를 `validate_report` 이슈로 기록하고 오프라인 테스트에서만 `ALLOW_UNCHECKED_PDF=1`로 경고로 낮췄다. (`reporting/sections.py`)
- `페이지 여유`: 상한 10p에 딱 맞추면 실제 실행마다 분량이 조금만 늘어도 재작성 루프만 소모하므로, 본문 조판(9.0pt/13.2pt, 상하 여백 20mm)과 보고서 프롬프트의 필드별 분량 예산으로 목표 9p를 두고 9p 초과는 경고로 남겼다(이전 실행 보고서 10p → 9p). (`reporting/export.py`, `agents/report.py`)
- `미사용 프롬프트`: LLM을 쓰지 않는 Supervisor용 `01_master_agent.md`와 대체된 `08_result_validator.md`를 프롬프트 레지스트리에 남기면 실제로 LLM 라우터가 있는 것처럼 오해되므로, 01은 `docs/SUPERVISOR_POLICY.md` 규칙 명세로 옮기고 08은 삭제했다. (`prompts.py`)
- `이전 실행물 분리`: 이전 master 구조 실행물(`final_state·validation·run_logs`)을 `outputs/`에 그대로 두면 새 Supervisor 실행 증빙으로 오인되므로, `outputs/legacy_rag/`로 옮기고 `RAG-Output_*` 제출물만 제자리에 두었다. (`outputs/legacy_rag/`)
- `LangSmith 연결`: 그래프 노드만 트레이스하면 노드 안의 LLM 호출 비용·지연을 볼 수 없으므로, 환경변수 트레이싱에 더해 OpenAI 클라이언트를 `wrap_openai`로 감싸 같은 run 아래에 남겼다. (`llm.py::_traced`)
- `그래프 이미지`: `draw_mermaid_png`만 쓰면 mermaid.ink에 접근할 수 없는 환경에서 이미지가 생성되지 않으므로, 실패 시 컴파일 그래프의 실제 노드·엣지를 Pillow로 그리고 Mermaid 원문도 함께 저장했다. (`scripts/export_graph.py`)
- `--report-only`: 별도 루프 코드로 보고서만 다시 돌리면 그래프와 재시도 규칙이 갈라지므로, 같은 Supervisor 그래프에 관점 재조사 한도 0인 Policy를 넣어 재사용했다. (`app.py::report_only`)
- `오프라인 테스트`: 실제 API로 흐름을 검증하면 키가 필요하고 실행마다 결과가 달라 재현이 안 되므로, 스키마 종류로 에이전트를 구분해 실행 회차별로 부족·편향·예외·중단을 주입하는 Fake LLM·Web·RAG를 만들었다. (`tests/fakes.py`)
