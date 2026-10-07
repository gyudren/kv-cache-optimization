# 설계 결정 노트

Supervisor 패턴으로 옮기면서 정한 것과 그 이유를 적는다. 항목 끝의 백틱은 코드 위치다.

## 패턴·라우팅

- 오케스트레이션 — 단계별 게이트 체인은 실행 순서가 엣지에 묶여 부족한 관점만 다시 부를 수 없으므로, 단일 Supervisor가 State를 보고 다음 노드를 고른다. `graph.py`, `supervisor/`
- Supervisor 판단 방식 — LLM 라우터는 같은 State에서도 결과가 달라져 재현·테스트가 안 되고 매 진입마다 호출 비용이 붙으므로, 라우팅 입력이 모두 구조화 필드인 점을 살려 결정적 규칙 함수로 둔다. `supervisor/policy.py::decide`
- 라우팅 지점 — conditional edge는 `supervisor` 하나뿐이고, 결정 사유를 남기기 위해 supervisor 노드가 `next_agents`·`last_decision`을 쓰고 라우터는 `next_agents`만 읽는다. `graph.py`, `supervisor/router.py`
- 할당 방식 — 정적 병렬 엣지는 항상 같은 묶음만 돌리므로, 미수집·부족 관점만 골라 `Send`로 동적 fan-out하고 페이로드는 관점 에이전트가 읽는 필드로 제한한다. `supervisor/router.py::route`
- 충분성 기준 — 세부 미확인 항목까지 부족으로 보면 매번 재시도 한도를 다 쓰므로, 판정을 막는 필수 결함(`missing`)과 고유 출처 수만 재조사 사유로 쓰고 `missing_optional`은 보고서 한계점에만 남긴다. `supervisor/policy.py::classify`
- 출처 집계 범위 — 시도별 출처를 누적하면 매 시도 1개씩만 찾아도 통과하므로, guard가 기록한 최신 시도의 `source_units`만 센다. `supervisor/guard.py`, `evaluation/quality.py::evidence_shortfalls`
- 재작업 지시 — 부족 사유 문장은 검색어로 쓸 수 없으므로 사유(`missing`)와 기술별 검색 힌트(`queries_by_tech`)로 나누고, 지시의 지문을 RAG 캐시 키에 넣어 겨냥한 질문만 다시 검색한다. `supervisor/policy.py::_rework`, `rag/workflow.py::answer_with_cache`
- 평가 노드 호출 — `report → quality_evaluator` 고정 엣지는 빈 보고서까지 평가하므로, 보고서도 Supervisor로 돌아오고 본문이 있을 때만 `evaluate`로 평가 노드를 부른다. `supervisor/policy.py::_report_and_eval_step`

## State

- 제어·페이로드 분리 — 라우팅이 결과 본문 구조에 의존하지 않도록, Supervisor는 제어 필드와 결과의 `sufficient`·`missing` 같은 신호만 읽는다. `state.py::GraphState`
- reducer — `Send`로 함께 할당된 노드가 같은 superstep에 같은 키를 쓰면 충돌하므로, dict 필드는 키 단위로 병합하고 `evidence`는 같은 에이전트의 이전 근거를 지운 뒤 중복 없이 이어 붙인다(`gaps`도 중복 제거). `state.py`
- 근거 공백 — 관점·기술·종류·내용을 가진 구조화 dict로 두어 보고서 7장과 편향·커버리지 규칙이 같은 데이터를 읽는다. `state.py::Gap`
- State 크기 — 결정 로그 본문은 `outputs/decisions_{trace_id}.jsonl`, RAG 캐시와 발췌 원문은 `data/cache/{trace_id}/`에 두고 State에는 `last_decision`과 160자 축약본만 남겨 체크포인트가 커지지 않게 한다. `observability.py`, `evidence_store.py`
- 상관 키 — uuid4 `trace_id` 하나를 LangGraph `thread_id`·LangSmith metadata·결정 로그 파일명에 함께 써서 손으로 맞출 일을 없앤다. `observability.py::run_config`
- 재개 — 메모리 체크포인터는 프로세스가 죽으면 사라지고 Postgres는 CLI에 과하므로, 파일 하나로 끝나는 `AsyncSqliteSaver`에 저장하고 `--resume`으로 이어 간다. `app.py::open_checkpointer`

## 실패 처리·품질 평가

- 에이전트 예외 — 예외가 그래프 밖으로 나가면 보고서가 남지 않으므로, 래퍼가 `node_status=failed`·`last_error`로 바꾸고 Supervisor가 한도 안에서 재시도한다. `supervisor/guard.py`
- 평가 방식 — 형식 검사만으로는 근거가 주장을 뒷받침하는지 볼 수 없고 Judge만 쓰면 판정이 흔들리므로, 규칙 검사를 하드 게이트로 두고 LLM Judge는 규칙 실패를 뒤집지 못하게 한다. `evaluation/quality.py::combine`
- 미달 경로 — 코드가 에이전트 판정을 옮긴 표는 보고서를 다시 써도 같은 값으로 채워지므로, 항목이 아니라 원인(`target_agents`)으로 나눠 관점이면 재조사하고 `report`면 재작성한다. `supervisor/policy.py::_report_and_eval_step`
- 후속 재조사 한도 — 종합·평가 요청을 충분성 재조사와 같은 한도로 세면 그 전에 한도가 바닥나 재조사 경로가 실행되지 않으므로, `FOLLOWUP_LIMITS`를 따로 두고 `excluded` 관점도 대상에 넣는다. `config.py`, `state.py::followup_counts`
- 재시도 상한 — 0~1회면 일시적 부족도 공백이 되고 3회 이상이면 같은 자료만 반복 검색하므로, 관점 2회·후속 1회·종합 1회·보고서 2회·평가 노드 1회로 둔다. `config.py::RETRY_LIMITS`
- 단계 상한 — `recursion_limit`만 두면 예외로 죽어 보고서가 없으므로, `MAX_STEPS`를 넘으면 조사를 멈추고 공백을 기록한 뒤 종합·보고서·평가까지 마치고 종료한다. `supervisor/policy.py::decide`
- 출처 단위 — 논문을 (문서, 페이지)로 세면 논문 한 편이 '고유 출처 2개'가 되므로 문서 단위로 세고, 기술당 원문 1편이 설계인 도메인 평가는 이 규칙 대신 단일 문헌 '적합'을 '조건부'로 낮춘다. `evidence_store.py::source_unit`, `agents/domain.py::downgrade_single_document`
- 편향 임계값 — 고유 출처 1개면 단일 기사로 판정이 굳고 3개 이상이면 자료가 적은 ITME가 늘 공백이 되므로 `MIN_DISTINCT_SOURCES=2`, 단일 발행처 비중은 3개 중 2개부터 걸리는 `0.6`으로 둔다. `config.py`
- [D] 사용 범위 — 설계 문서를 어디서나 인용으로 인정하면 허용 문서 풀 밖 자료가 기술 사실의 근거가 되므로, SUMMARY·1·2장의 설계 조건에만 허용하고 그 밖의 [D] 단독 인용은 규칙 위반이다. `evaluation/quality.py::check_groundedness`
- 중립성 규칙 — 단어 하나로 잡으면 면책 문장을 오탐하고 문장 어디든 부정어만 있으면 면제하면 앞 절의 우열을 놓치므로, 두 기술 간 비교·추천·지시 표현만 잡고 같은 절의 바로 뒤 부정만 면제한다. `evaluation/quality.py::neutrality_issues`
- 근거 공백 인정 — 한도를 넘긴 공백까지 실패로 두면 공개 근거가 없는 경우 보고서가 영원히 미검증이므로, Supervisor가 기록한 관점·기술에 한해 '결과 없음'만 면제하고 절 구조는 항상 요구한다. `evaluation/quality.py::_acknowledged`

## 실행·저장

- 비동기 실행 — 작업 노드는 `async def`이고 동기 SDK 호출은 `asyncio.to_thread`로 넘기며, 에이전트 안의 질문 병렬 처리는 contextvars를 복사하는 `ContextThreadPoolExecutor`를 써서 LLM·웹 호출이 LangSmith 그래프 run의 자식으로 남게 한다. `supervisor/guard.py`, `agents/technology.py`
- 실행 동시성 — 4관점을 동시에 돌리면 메모리가 관점 수만큼 커지고 MPS 임베딩 모델 경합으로 프로세스가 abort되므로, 할당은 한 번에 하되 실행은 `AGENT_CONCURRENCY`(기본 1)로 하나씩 하고 질의 임베딩은 락으로 직렬화한다. `observability.py::run_config`, `rag/retrieve.py`
- 문서 임베딩 캐시 — 실행마다 전체 청크를 다시 임베딩하면 시작에 수십 초가 걸리므로, 모델 ID와 청크로 만든 지문을 키로 `data/cache/index/`에 한 번 저장하고 청크나 모델이 바뀌면 자동으로 다시 만든다. `rag/index.py::load_or_embed`
- 페이지 검사 — 글자 수 추정은 표 비중에 따라 오차가 크므로 내보내기와 같은 reportlab 조판으로 실제 페이지를 세고, 목표 9p 초과는 경고, 상한 10p 초과는 이슈로 남긴다. `reporting/export.py::report_page_count`
- 오프라인 테스트 — 실제 API로는 키가 필요하고 재현이 안 되므로, 회차별로 부족·편향·예외를 주입하는 Fake LLM·Web·RAG로 흐름을 검증한다. `tests/fakes.py`
