# QA 보고서 (feat/multi-agent-supervisor)

> **최신 판정: 반복 3, 87/100. 코드는 로컬 실제 실행에 들어갈 준비가 됐다(남은 항목은 D-01 실제 실행과 캡처뿐이며, 실행 후 최대 100점).** 상세 내용은 문서 끝의 "반복 3"을 본다. 반복 1·2 기록은 수정하지 않고 보존한다.

# 반복 1

- 검증자: QA Senior · 검증일 2026-10-07 · 대상 커밋 `1896b1c..62479cc` (E1~E14)
- 기준: Notion「(참고) 평가 항목」 100점 루브릭, `docs/DEV_PLAN.md`, `docs/DECISIONS.md`, `README.md`
- 환경 제약: 이 세션에는 실제 `OPENAI_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_API_KEY`가 없다. 실제 실행과 LangSmith 캡처는 **미검증**이며 만점을 주지 않았다. 코드가 그 실행을 할 준비가 됐는지는 따로 판정했다.
- 원칙: 소스 코드는 수정하지 않았다. 검증 스크립트는 모두 저장소 밖 scratchpad에서 실행했다. 저장소 `outputs/`는 바뀌지 않았다(`git status` clean).

## 1. 총점

**70 / 100. 만점이 아니므로 Engineer에게 반려한다.**

| 항목 | 배점 | 점수 | 판정 요약 | 핵심 증거 |
|---|---|---|---|---|
| 패턴 적용 정합성 | 20 | **15** | Supervisor 단일 conditional edge, `Send` 동적 fan-out, 부족 관점만 재작업, 실패 fallback, 종료 판정이 모두 구현됐다. 감점 사유: ① `report → quality_evaluator` 고정 엣지 때문에 report가 Supervisor로 돌아가지 않고, 실패한 보고서도 평가된다(D-03). ② tech·domain 재작업 지시가 해당 에이전트에 실제로 전달되지 않는다(D-04). ③ "규칙 기반 Supervisor vs LLM 라우터"를 고른 이유가 없다(D-06). | `graph.py:42,46`, `supervisor/policy.py:43-52,201-278`, 엣지 덤프(§3.1) |
| 동적 동작 실증 | 20 | **11** | Fake 시나리오에서 라우팅 경로가 State에 따라 달라지고, `retry_counts`와 사유가 결정 로그에 남는다. 감점 사유: 실제 LangSmith 트레이스와 실행 산출물이 없다(미검증, D-01). 에이전트 내부 스레드풀에서 LangSmith 부모 run이 끊긴다(D-02). | `tests/test_scenarios.py`, §3.2·§3.6 |
| State Schema 설계 | 20 | **17** | 7항목 모두 README(`README.md:91-98`)와 코드(`state.py`)에 대응하고, reducer 병렬 병합 테스트가 통과한다. 감점 사유: "지속성 비용" 근거가 실측과 맞지 않는다(D-05). | `state.py:21-70`, `tests/test_graph_structure.py:63-82`, §3.5 |
| 품질 평가 노드 | 15 | **12** | 보고서 뒤에 독립 노드가 있다. 4항목을 Hybrid(규칙+Judge)로 항목별 판정하고, 규칙 실패는 Judge가 뒤집지 못한다. 미달하면 원인별로 루프를 돈다(테스트로 확인). 감점 사유: 실패한 보고서도 평가한다(D-03). 근거 공백이 있으면 구조 결함까지 면제한다(D-08). 폰트가 없으면 10p 검사를 조용히 건너뛴다(D-09). 평가 기반 재조사가 tech·domain에서는 실효가 없다(D-04). | `evaluation/quality.py:158-262`, §3.2 |
| 코드 구조·모듈 분리 | 5 | **4** | `supervisor/`·`evaluation/`(조정)과 `agents/`(작업자)가 분리됐고 import 금지 테스트가 있다. README 트리는 실제 트리와 일치한다. 감점 사유: 삭제된 master 역할의 잔재가 남아 있다(D-11). | `tests/test_quality_and_app.py:191-195`, §3.7 |
| 실행 결과 재현성 | 10 | **6** | `pytest` 26/26 통과. 항상 실패하는 Fake 7종이 모두 유한 스텝 안에 END에 닿는다. 모든 모듈이 import되고 CLI 4개 경로가 연결돼 있다. 감점 사유: 실제 실행 결과가 없다(D-01). `outputs/`가 이전 master 실행물이라 제출 트레이스와 맞지 않는다. `--export-only`가 기존 로그를 덮어쓴다(D-07). | §3.3·§3.4·§3.8 |
| Output 보고서 | 10 | **5** | 코드가 SUMMARY·REFERENCE 필수 목차, 10p 상한, `Agent_판교_9반_…` 파일명을 강제한다. 그러나 새 파이프라인이 만든 Agent 보고서가 `outputs/`에 없다(미검증, D-01). 이전 보고서를 내보내면 정확히 10p라 여유가 없다(D-15). | `reporting/sections.py:44,122-163`, `config.py` `REPORT_STEM` |
| **합계** | **100** | **70** | | |

## 2. Engineer의 질문에 대한 판정: `report → quality_evaluator` 고정 엣지

**결론: 고정 엣지를 없애고 `report → supervisor`로 바꿀 것을 권고한다. 지금 구조로는 루브릭 점수를 잃을 위험이 있다.**

1. **루브릭 문구와 충돌한다.** Supervisor 필수 항목은 "하위 에이전트는 Supervisor와만 통신"이다. `report`는 `AGENT_NODES`(`graph.py:21`)에 속한 작업자인데, 엣지를 덤프하면 `report -> quality_evaluator fixed`가 그대로 보인다(§3.1). 평가자가 엣지를 기계적으로 검사하면 "보고서 에이전트가 Supervisor를 거치지 않고 평가 노드로 직접 넘긴다"로 읽힌다. 평가 노드를 "Supervisor 측 게이트"로 분류한 것은 문서상 주장일 뿐이고, 그래프에서는 별도 노드다.
2. **결정 로그에 평가 결정이 남지 않는다.** 정상 실행의 Supervisor 결정은 `dispatch → synthesis → report → end:passed`다(`tests/test_scenarios.py:247`). "근거와 보고서를 확인하고 평가로 보낸다"는 Supervisor 결정이 트레이스에 없어 동적 동작 실증이 약해진다.
3. **실제 결함이 있다(D-03).** report가 예외로 `failed`가 되어도 고정 엣지가 평가 노드를 실행한다. 빈 보고서에 LLM Judge를 호출하고, `fail:groundedness,coverage`라는 잘못된 평가 로그를 남긴다.
4. **바꿔도 안전하다.** DECISIONS가 걱정한 "평가 없이 종료하는 경로"는 정책에 이미 막혀 있다. `policy.py:219-221`이 보고서는 `done`이고 평가는 `pending`인 상태에서 항상 `evaluate`를 고른다. `supervisor → quality_evaluator` conditional 엣지도 이미 있다(`graph.py:42`). 예외는 `end:hard_limit`(`policy.py:285`) 하나다. 마무리 시퀀스(종합 → 보고서 → 평가 → 종료)는 `FINALIZE_STEPS=4`에 딱 맞으므로, 여유분으로 5로 올릴 것을 권고한다.
5. **함께 바꿀 것:**
   - `graph.py:43-46`
   - `tests/test_graph_structure.py:26-30`: 모든 작업자의 후속 노드가 `{supervisor}`인지 검사
   - 새 불변식 테스트: 마지막 `report` 이후에 `quality_evaluator`가 실행되지 않았다면 `end:passed`가 될 수 없다
   - `docs/DECISIONS.md:32` "평가 노드 위치"의 이유 문장
   - README 다이어그램(`REPORT --> EVAL`)

## 3. 검증 수행 내역 (명령 → 결과)

### 3.1 그래프 컴파일과 엣지 덤프
`build_graph(None,None,None).get_graph().edges`
```
__start__->supervisor fixed | tech/market/stakeholder/domain/synthesis -> supervisor fixed
report -> quality_evaluator fixed | quality_evaluator -> supervisor fixed
supervisor -> {tech,market,stakeholder,domain,synthesis,report,quality_evaluator,__end__} cond
branches = {'supervisor': ['route']}
```
- 하위 에이전트 사이의 직접 엣지는 0개다. conditional 분기는 Supervisor 1곳뿐이다. 예외는 `report → quality_evaluator` 하나다(§2).
- `outputs/architecture.png`와 `.mmd`는 위 엣지와 일치한다(`scripts/export_graph.py`로 다시 생성해 비교했고 내용 차이는 없었다).

### 3.2 Fake 시나리오: 라우팅과 재작업
`python -m pytest -q` → **26 passed in 3.87s**

직접 만든 고장 주입 스크립트(`adv.py`)를 `stream_mode="values"`로 실행한 결과:

| 시나리오 | 종료 상태 | Supervisor 진입 / superstep | 결정 경로 요약 |
|---|---|---|---|
| 모든 노드가 항상 예외 | `unverified` | 9 / 21 | dispatch×3 → synthesis → retry:synthesis → report → retry:report×2 → `end:report_failed` |
| report 1회 예외 | `completed` | 5 / 12 | report → **quality_evaluator `fail:groundedness,coverage` (빈 보고서 평가)** → retry:report → pass |
| Judge가 항상 예외 | `unverified` | 5 / 11 | report → retry:quality_evaluator → `end:unverified` |
| Judge 4항목 항상 미달(market 지목) | `unverified` | 12 / 29 | reinvestigate:market×2 → rewrite:report×2 → `end:unverified` |
| 시장·이해관계자 판정이 항상 한쪽 | `completed_with_gaps` | 11 / 26 | reinvestigate:market,stakeholder×2 → 공백 기록 → rewrite:report → pass |
| 4관점 항상 부족 + 금지 표현 | `unverified` | 8 / 19 | dispatch×3 → synthesis → report → rewrite×2 → `end:unverified` |
| 종합 항상 예외 | `completed_with_gaps` | 5 / 11 | synthesis → retry → skipped(공백) → report → pass |

- 7개 시나리오 모두 유한 스텝(superstep 29 이하)에서 END에 닿았다. `RECURSION_LIMIT = 82`에 한참 못 미친다. 종료 보장은 **충족**한다.
- 같은 입력이라도 State(부족, 예외, 평가 미달)에 따라 경로가 달라진다. 재작업 횟수는 `retry_counts`와 결정 로그 사유(`근거 부족 재조사 1/2`)에 남는다. **충족.**

### 3.3 재개 (SqliteSaver)
`resume.py`: 크래시를 주입한 뒤 같은 sqlite 파일로 그래프를 다시 만들어 `invoke(None)`을 호출했다.

| 크래시 지점 | snapshot.next | 재개 후 재실행된 노드 |
|---|---|---|
| market(병렬 fan-out 중) | `('market',)` | market만 다시 실행. tech·stakeholder·domain은 1회 그대로 |
| quality_evaluator | `('quality_evaluator',)` | judge만 다시 실행 |
| report | `('report',)` | report만 다시 실행 |

재개/복구는 **충족**한다. 병렬 superstep 중 끝난 Send의 pending write도 보존됐다.

### 3.4 app.py 경로 (키 없음, `OUTPUT_DIR`은 임시 폴더)

| 명령 | 결과 |
|---|---|
| `python app.py` | `RuntimeError: OPENAI_API_KEY and TAVILY_API_KEY are required`, exit 1 (의도대로) |
| `--resume abc` | 같은 오류, exit 1 |
| `--report-only` | `OPENAI_API_KEY is required`, exit 1 |
| `--export-only` (이전 `final_state.json`) | Agent 파일명으로 md·pdf 생성(10p), `품질 평가 결과 없음`으로 exit 2. **`run_logs.json`이 `[]`로 덮어써짐(D-07)** |
| `--resume x --export-only` | argparse 상호 배제 오류 (의도대로) |

`kv_eval` 하위 모든 모듈을 import했고 실패는 0건이다. `eval/`과 `scripts/`에서 삭제된 `agents/master.py`를 import하는 곳은 없다. `scripts/validate_prompt_package.py` → OK.

### 3.5 State 크기와 지속성
- 이전 `outputs/final_state.json`의 필드별 크기: `evidence` **480,596B**(320건, 발췌 최대 1,508자), `tech_result` 85,542B, `domain_result` 88,952B(대부분 rag_cache).
- 새 구조는 rag_cache를 디스크로 옮겨 약 170KB를 줄였다. 하지만 **evidence 약 480KB는 그대로 State에 남는다.** 발췌 상한 1,600자는 실데이터(최대 1,508자)에서 아무것도 자르지 않는다(D-05).
- Fake 실행(재작업 많음) 한 번에 checkpoint 27개, `checkpoints.checkpoint` 합계 **1.86MB**가 쌓였다. 체크포인트마다 State 전체가 직렬화된다. 실데이터 기준으로 추정하면 실행 1회에 10MB를 넘는다.
- 무한 증가는 아니다. evidence는 dedup 후 재시도 상한 안에서만 늘고(75 → 203건), gaps·feedback·last_error는 키 단위로 덮어쓴다. 다만 이전 시도의 evidence는 지워지지 않는다.

### 3.6 LangSmith 상관
`run_config`가 `thread_id`, `metadata.trace_id`, `tags`, `run_name`을 설정한다(`observability.py`). **충족.** 하지만 다음 실험으로 결함을 확인했다.
```
@traceable 노드 안에서 ThreadPoolExecutor로 get_current_run_tree() 호출
→ main run: True | worker sees parent: False
```
`technology.py:67`과 `domain.py:64`의 스레드풀에서 실행되는 RAG LLM 호출(질문 27개 × plan·judge·rewrite·answer)과 웹 검색은 그래프 run 밖의 **고아 root run**이 되고 `trace_id` metadata도 붙지 않는다(D-02).

### 3.7 README 디렉터리 구조와 실제 트리
README 트리의 경로(`.env.example`, `supervisor/`, `evaluation/quality.py`, `tools/`, `rag/cache.py`, `scripts/`, `docs/` 등)는 모두 실제로 존재한다. 불일치는 다음과 같다.
- README는 `prompts/`를 "역할별 프롬프트(01~08)"라고 설명한다. 그런데 `01_master_agent.md`와 `08_result_validator.md`는 런타임에 쓰이지 않는다(D-11).
- 트리에 `.claude/`와 `prompts/README.md`가 빠져 있다(무시할 수준).

### 3.8 결정 이유 한 줄 ("대안이면 어떻고, 그래서 이걸 선정했다")
- DECISIONS.md에는 형식에 맞는 문장이 26개 있다. README에는 패턴 선정 이유와 State 7항목의 이유가 있다.
- **빠진 결정:**
  - 규칙 기반(결정적) Supervisor 정책 vs LLM 라우터 Supervisor
  - 재시도 상한 값(2/1/2/1), `MAX_STEPS=20`, `FINALIZE_STEPS=4`
  - `MIN_DISTINCT_SOURCES=2`, `MAX_SINGLE_SOURCE_SHARE=0.6`
  - `Send(name, state)`로 State 전체를 전달하는 방식
- README에는 Hybrid 평가 방식, 평가 노드 위치, 체크포인터 선정의 이유 문장이 없고 DECISIONS로 위임만 한다(D-06).

## 4. 결함 목록

| ID | 심각도 | 루브릭 항목 | 증거 | 재현 방법 | 기대 동작 |
|---|---|---|---|---|---|
| D-01 | **Blocker** | 동적 동작 실증, 재현성, Output | `outputs/`에 `Agent_*.md/pdf`, `decisions_*.jsonl`, LangSmith 캡처(`tracing-*.png`)가 없다. `outputs/run_logs.json`, `final_state.json`, `validation.json`은 이전 master 구조의 실행물(`master_init`, `master_report_gate` 노드)이다. | `grep -c master_ outputs/run_logs.json`, `ls outputs/` | 키가 있는 로컬에서 `python app.py`를 1회 실행해 새 산출물로 교체한다. 같은 `trace_id`의 LangSmith 트레이스 캡처, 결정 로그, 보고서를 함께 제출하고, README 실행 예시와 대조한다. **코드는 실행 준비 상태지만 D-02와 D-03을 먼저 고치고 실행할 것.** |
| D-02 | Major | 동적 동작 실증, State(상관) | `agents/technology.py:67`, `agents/domain.py:64`의 `ThreadPoolExecutor`는 contextvars를 넘기지 않는다. | §3.6 실험: 워커 스레드에서 `get_current_run_tree()`가 None | 노드 안의 모든 LLM·웹 호출이 그래프 run 아래 자식 run으로 남고 `trace_id` metadata를 갖는다(`langsmith.utils.ContextThreadPoolExecutor` 또는 `contextvars.copy_context().run` 사용). |
| D-03 | Major | 패턴 정합성, 품질 평가 노드 | `graph.py:46`의 `add_edge("report","quality_evaluator")`. 보고서가 실패해도 평가가 실행된다. | `FakeLLM(raise_on={"report":1})`: 로그가 `report → quality_evaluator fail:groundedness,coverage(필수 목차 0회) → retry:report`이고 judge가 2회 호출됨 | report도 Supervisor로만 돌아간다. 평가는 Supervisor의 `evaluate` 결정(`policy.py:219-221`)으로만 실행되고, 실패한 보고서에는 Judge를 부르지 않는다(§2). |
| D-04 | Major | 패턴 정합성(재작업 요청), 품질 평가 노드(루프 실효성) | `rag/workflow.py:58-68`의 `answer_with_cache`는 이전에 sufficient였던 답을 재사용한다. `technology.py:103-113`과 `domain.py:73-77`의 LLM 프롬프트에는 `feedback`(missing)이 들어가지 않는다. | `FakeLLM(judge_fail={"bias_control":(1,"domain")})` + 호출 수를 세는 RAG: `reinvestigate:domain`이 실행됐는데 RAG 호출 27회 중 피드백이 붙은 호출은 **0회**이고, domain의 새 검색도 0회 | 재작업 지시(`missing`, 재검색 질의)가 대상 에이전트의 검색과 프롬프트에 실제로 반영된다. 평가 기반 재조사에서는 지목된 기술·차원의 캐시를 무효화하거나, 반대 방향 질의를 추가 검색한다. 이를 검증하는 테스트도 추가한다. |
| D-05 | Major | State Schema(지속성 비용) | `state.py:40`의 `EXCERPT_MAX_CHARS=1600`은 실데이터 최대 1,508자를 자르지 못한다. evidence 480,596B가 State에 남는다. Fake 실행 checkpoint 합계 1.86MB. | §3.5 측정 스크립트 | README·DECISIONS가 주장한 "대용량은 외부, State에는 참조"와 실제가 일치한다. 예: 발췌 원문은 디스크(`source_id` 키)에 두고 State에는 claim·citation·짧은 발췌(약 300자)만 둔다. 또는 실측 크기를 README에 적고 근거 문장을 고친다. |
| D-06 | Major | 패턴 정합성(라우팅 방식), 전 항목(결정 이유 요구) | DECISIONS와 README에 "규칙 기반 정책 vs LLM 라우터"의 이유가 없다. 수치 상한과 임계값의 이유도 없다(§3.8). `prompts/01_master_agent.md`는 LLM 라우터용 출력 형식을 정의하지만 실제로는 쓰이지 않는다. | `grep -n "LLM" docs/DECISIONS.md` | 빠진 결정마다 "대안이면 어떻고, 그래서 이걸 선정했다" 한 줄을 DECISIONS와 README에 둘 다 쓴다. |
| D-07 | Minor | 재현성 | `app.py:138-139`는 decisions가 비어 있어도 `run_logs.json`을 `[]`로 덮어쓴다. README는 "이전 과제 산출물 수정하지 않음"이라고 적었다. | `OUTPUT_DIR=<복사본> python app.py --export-only` → `run_logs.json` 2바이트 | 결정 로그가 없으면 기존 파일을 보존한다(또는 별도 파일명으로 쓴다). |
| D-08 | Minor | 품질 평가 노드(관점 커버리지, 편향 통제) | `quality.py:178`의 `if problems and not _acknowledged(gaps, name)`은 관점 단위 gap 하나로 "판정 표 없음", "서술 없음"까지 면제한다. `quality.py:67`은 `name:` 접두 gap 하나로 두 기술의 편향 검사를 모두 면제한다. | gaps에 `market: …`를 넣고 4.2 절 본문을 지운 보고서로 `check_coverage` → 통과 | 근거 공백은 "결과 없음"만 면제한다. 절 구조(판정 표, 근거 부족 서술)는 항상 요구한다. |
| D-09 | Minor | 품질 평가 노드(Groundedness), Output | `reporting/sections.py:136-141`은 한글 폰트가 없으면 10p 검사와 1/2p 검사를 이슈 없이 건너뛴다(`page_check: skipped`). | 폰트가 없는 환경에서 `validate_report` | 검사를 건너뛰면 평가 결과에 미검사로 드러내거나 groundedness 규칙을 실패 처리한다. |
| D-10 | Minor | 패턴 정합성(재작업 질의) | `policy.py:72`의 `queries or issues`. Judge만 미달이면 사유 문장(`"bias_control: bias_control 판정"`)이 그대로 재검색 질의가 된다. | §3.2의 cache 스크립트 출력에서 `domain feedback.rewritten_queries` 확인 | DECISIONS "평가 기반 재조사 질의"대로 힌트 질의만 쓴다(사유 문장으로 검색하지 않는다). |
| D-11 | Minor | 코드 구조 | `prompts.py:6,13`이 `01_master_agent.md`와 `08_result_validator.md`를 등록하지만 호출하는 곳이 없다. prompts 02~07에 "### Master 피드백"과 "재시도는 Master가 관리"가 남아 있다. `tools/__init__.py:5`, `rag/workflow.py:61`의 docstring도 "Master"다. `llm.py:1,46`은 "GPT-5.6 Sol"이고 `MODEL_ID`는 `gpt-5.6-terra`다. | `grep -rn -i master src prompts` | 삭제된 역할의 이름을 Supervisor로 정리한다. 쓰지 않는 프롬프트는 설계 문서로 옮기거나 README에 "실행 미사용(설계 명세)"로 표시한다. |
| D-12 | Minor | 재현성(문서) | `.env.example`에는 "Tavily 선택, DuckDuckGo로 대체"라고 적혀 있다. 실제로는 `config.py require_credentials`가 Tavily를 필수로 요구하고, DuckDuckGo 구현도 없다. | `grep -i duck src -r` → 없음 | 문서와 코드를 일치시킨다. |
| D-13 | Minor | 재현성 | `app.py:196-198`: `--report-only`가 `configure_tracing()`보다 먼저 LLM(wrap_openai)을 만든다. `app.py:97-102`: `--resume`이 체크포인트 존재를 확인하기 전에 색인부터 만든다(잘못된 trace_id면 수 분을 낭비). `--query`는 `--resume`과 함께 쓰면 무시된다. | 코드 리딩 | 트레이싱을 설정한 뒤 LLM을 만든다. 체크포인트를 확인한 뒤 런타임을 만든다. |
| D-14 | Minor | 패턴 정합성(근거 충분성 판단 주체) | `policy.py:43-52`의 Supervisor 충분성 판단은 에이전트가 스스로 보고한 `sufficient` 플래그만 읽는다. 결정적 출처 임계(`MIN_DISTINCT_SOURCES`)는 보고서를 쓴 뒤에야 적용된다. | 코드 리딩 | "Supervisor가 근거 충분성을 평가한 뒤 보고서"를 더 분명히 하려면, 종합 전에 Supervisor가 관점×기술별 근거 수와 판정 공란을 결정적으로 검사한다. |
| D-15 | Minor | Output 보고서 | 이전 보고서를 `--export-only`로 내보내면 정확히 10p다(상한). | §3.4 | 실제 실행 보고서가 9p 이하로 여유를 두는지 확인한다(초과하면 재작성 루프만 소모한다). |

## 5. 판정과 Engineer 수정 우선순위

**판정: 만점이 아니므로 반려한다(70/100).** 오프라인 기준으로 코드의 골격은 견고하다. Supervisor 단일 라우터, Send fan-out, reducer, 3중 종료 가드, Hybrid 평가, SQLite 재개가 모두 동작한다. 남은 감점의 절반은 실행 증빙이 없는 데서 오고, 나머지 절반은 재작업과 관측 경로가 실제로는 끊기는 결함에서 온다.

1. **D-03**: `report → supervisor`로 바꾸고(§2 지침), 불변식 테스트를 추가하고, `FINALIZE_STEPS=5`로 올린다. DECISIONS, README 다이어그램, 엣지 테스트를 같이 고친다.
2. **D-04**: tech·domain 재작업 시 feedback을 프롬프트와 재검색에 반영하고, 평가 기반 재조사에서는 캐시를 우회한다. 테스트를 추가한다.
3. **D-02**: 스레드풀의 LangSmith 컨텍스트를 전파한다(`ContextThreadPoolExecutor`).
4. **D-05**: evidence 발췌를 외부에 저장하거나 축약하고, README의 지속성 근거를 실측치로 바꾼다.
5. **D-06**: 빠진 결정 이유 한 줄을 DECISIONS와 README에 추가한다(LLM 라우터 대비, 수치 상한, 임계값).
6. **D-01**: 위 수정이 끝난 뒤 키가 있는 로컬에서 실제 1회 실행한다. 새 `Agent_*.md/pdf`, `validation.json`, `decisions_{trace_id}.jsonl`, `run_logs.json`, `final_state.json`, LangSmith 캡처(같은 trace_id)를 커밋하고, 이전 master 실행물은 정리한다.
7. Minor(D-07~D-15)는 같은 반복에서 함께 정리한다.

다음 QA 반복에서는 위 항목을 다시 검증하고, 실제 실행 산출물을 결정 로그·LangSmith 트레이스·보고서와 서로 대조한다.

---

# 반복 2

- 검증일 2026-10-07 · 대상 커밋 `7ed9f10..18bca72` (Engineer 커밋 9개)
- 원칙은 반복 1과 같다. Engineer 요약을 그대로 믿지 않고, 반복 1의 재현 스크립트(엣지 덤프, 고장 주입 7종, 캐시, 재개)를 새 코드로 다시 돌렸다. 회귀를 찾기 위한 스크립트도 새로 만들어 돌렸다: 병적 단계 상한, Send 페이로드, evidence 저장소 연결, app 경로 시뮬레이션. 모두 저장소 밖 scratchpad에서 실행했고 저장소는 `git status` clean 상태를 유지했다.
- 실제 API 실행과 LangSmith 캡처는 여전히 **미검증**이다(키 없음). 해당 배점은 보류했고, 실행이 끝나면 도달할 수 있는 최대 점수를 따로 적었다.

## R2-1. 총점

**84 / 100. 만점이 아니므로 반려한다.** 코드 결함 중 Major는 모두 해소됐다. 남은 큰 감점은 D-01(실제 실행·캡처 미제출) 하나이고, 나머지는 Minor 5건이다.

| 항목 | 배점 | 반복 1 | 반복 2 | 실행 후 최대 | 판정 요약 | 핵심 증거 |
|---|---|---|---|---|---|---|
| 패턴 적용 정합성 | 20 | 15 | **19** | 20 | 작업 노드 7개가 모두 `supervisor`로만 복귀하고, 평가는 Supervisor의 `evaluate` 결정으로만 실행된다. 재작업 지시가 RAG와 프롬프트에 실제로 들어간다. Supervisor가 결정적 충분성 검사를 한다. 남은 문제는 부족 사유 문장이 그대로 웹 질의가 되는 것(D-16)과 누적 근거로 충분성이 통과되는 것(D-17)이다. | `graph.py:43-46`, `policy.py:52-55,225-238`, R2-3.1·3.4 |
| 동적 동작 실증 | 20 | 11 | **14** | 20 | 결정 로그에 `evaluate`가 남는다. 워커 스레드도 부모 run을 이어받는 것을 실험으로 확인했다. 실제 LangSmith 트레이스는 미검증이다(D-01). | R2-3.2·3.6 |
| State Schema 설계 | 20 | 17 | **19** | 19 | 발췌 원문을 디스크로 옮겼다(이전 실행 evidence 491KB에서 228KB). Send 페이로드를 줄였지만 에이전트가 읽는 필드는 빠지지 않았다. 재개는 저장소 변경 후에도 정상이다. 남은 문제: 체크포인트마다 State 전체를 저장하는 비용이 남아 있다(D-19). 실행 후에도 Minor가 남으면 1점은 감점이 유지된다. | `evidence_store.py`, `router.py:23-37`, R2-3.5 |
| 품질 평가 노드 | 15 | 12 | **14** | 14 | 실패하거나 빈 보고서는 평가하지 않는다. 공백 면제 범위를 줄였다. 조판을 검사할 수 없으면 이슈로 남긴다. 평가 기반 재조사가 실제로 재검색한다. 남은 문제: 저장소가 없으면 원문 대신 300자 발췌로 조용히 판정한다(D-18). | `quality.py:66-87,175-196`, R2-3.3 |
| 코드 구조·모듈 분리 | 5 | 4 | **5** | 5 | master 잔재(01·08 프롬프트, 명칭)를 정리했다. README 트리는 실제 트리와 일치한다(`evidence_store.py`, `SUPERVISOR_POLICY.md`, `legacy_rag/`, scripts 포함). | `grep -rni master src prompts` 0건 |
| 실행 결과 재현성 | 10 | 6 | **7** | 10 | `pytest` 41/41 통과. 고장 주입 7종과 병적 단계 상한 시나리오가 모두 보고서를 남기고 정상 종료한다. app의 run, `--resume`, `--report-only`, `--export-only`를 Fake로 끝까지 실행했다. 실제 실행은 미검증이다(D-01). | R2-3.2·3.4·3.7 |
| Output 보고서 | 10 | 6→5 | **6** | 10 | 이전 보고서를 내보내면 9p다(목표 9p, 상한 10p). 새 파이프라인이 만든 `Agent_*` 보고서는 아직 없다(D-01). | R2-3.7 |
| **합계** | **100** | **70** | **84** | **97** | | |

- **실행 후 최대 97점**: D-01만 해결하면 보류한 13점(동적 6, 재현성 3, Output 4)이 회복되어 84 + 13 = 97점이 된다. Minor D-16~D-19까지 고치고 실행 결과가 아래 R2-5의 확인 기준을 통과하면 100점이 가능하다.

## R2-2. 반복 1 결함 재검증 결과

| ID | 반복 1 심각도 | 판정 | 재검증 근거 |
|---|---|---|---|
| D-01 | Blocker | **미해결(부분 진전)** | 이전 실행물은 `outputs/legacy_rag/`로 옮겼고 `scripts/capture_checklist.md`를 추가했다. 그러나 `Agent_*` 보고서, `decisions_*.jsonl`, LangSmith 캡처가 여전히 없다. |
| D-02 | Major | **해결** | `ContextThreadPoolExecutor`로 바꿨다(`technology.py:70`, `domain.py:66`). 실험 결과 `submit parent: True`, `map parent: True`. 테스트 `test_worker_threads_inherit_graph_run_context` 통과. |
| D-03 | Major | **해결** | 엣지 덤프에서 `report -> supervisor`이고 `report -> quality_evaluator`는 없다. report 1회 예외 시나리오에서 `report → retry:report → evaluate → pass` 순서로 judge가 1회만 호출된다(반복 1에서는 2회였고 빈 보고서도 평가했다). 테스트 3개(`end_passed_requires_evaluation…`, `failed_report_is_not_evaluated`, `report_always_failing…`) 통과. |
| D-04 | Major | **해결** | 반복 1과 같은 시나리오(`judge_fail bias_control→domain`)에서 RAG 호출은 41회이고 그중 **14회가 피드백 포함 재검색**이다(반복 1은 0회). market·stakeholder·tech·domain 프롬프트 끝에 `rework_note`가 붙는다. |
| D-05 | Major | **해결(잔여 Minor는 D-19)** | `measure_state_size.py`: 이전 실제 실행 evidence 491,320B를 축약본으로 바꾸면 227,639B다. Fake 재작업 실행의 체크포인트 21개 누적은 1,856,192B이고 외부 저장소는 231,667B다. 보고서와 Judge는 `hydrate()`로 원문을 받는다(Fake에서 발췌 900자 확인). |
| D-06 | Major | **해결** | DECISIONS에 43개 항목이 있다. README에 "설계 결정(선정 이유)" 절이 생겼다(LLM 라우터 대비, 평가 노드 위치, Hybrid, 체크포인터, 재시도 상한, MAX_STEPS, FINALIZE_STEPS, 편향 임계값, 충분성 판단 주체). |
| D-07 | Minor | **해결** | 이전 State로 `--export-only`를 실행해도 `run_logs.json`(12,967B)이 보존된다. |
| D-08 | Minor | **해결** | 반복 1 재현 그대로(4.2 절을 비우고 market gap 추가, excluded)인데 이제 `check_coverage`가 **False**를 반환한다. |
| D-09 | Minor | **해결** | 폰트가 없으면 이슈로 남긴다(`sections.py:163-167`). 테스트 환경에서만 `ALLOW_UNCHECKED_PDF=1`로 경고로 낮춘다. |
| D-10 | Minor | **해결** | Judge만 미달이면 질의가 `independent sources limitations adoption evidence`가 된다. 단, 같은 유형의 문제가 Supervisor 부족 사유에서 새로 나타났다(D-16). |
| D-11 | Minor | **해결** | `grep -rni master src prompts app.py scripts .env.example` 0건. 프롬프트 레지스트리에서 01·08을 제거했고 Supervisor 명세는 `docs/SUPERVISOR_POLICY.md`로 옮겼다. |
| D-12 | Minor | **해결** | `.env.example:11`에 "필수, 대체 검색 엔진 없음"이 명시됐다. |
| D-13 | Minor | **해결** | `--resume`이 색인을 만들기 전에 체크포인트를 확인한다. `--query`를 재개나 내보내기와 함께 쓰면 오류를 낸다. `--report-only`는 트레이싱 설정을 먼저 한다. |
| D-14 | Minor | **해결** | `classify`가 `evidence_shortfalls`로 다시 검사한다(`policy.py:52-55`). 단, 누적 근거로 통과되는 문제가 남았다(D-17). |
| D-15 | Minor | **해결** | 이전 State를 내보내면 PDF **9p**다(반복 1은 10p). |

## R2-3. 검증 수행 내역

### 3.1 엣지 덤프
```
tech/market/stakeholder/domain/synthesis/report/quality_evaluator -> supervisor (fixed, 7개)
supervisor -> {7개 작업 노드, __end__} (cond) · branches = {'supervisor': ['route']}
```
작업 노드 사이의 엣지는 0개이고, 모든 작업 노드의 후속 노드는 supervisor뿐이다. **충족.**

### 3.2 고장 주입 7종 재실행 (반복 1의 `adv.py`)

| 시나리오 | 반복 2 결과 | 비고 |
|---|---|---|
| 모든 노드 항상 예외 | `unverified`, 진입 9 / superstep 18, `end:report_failed` | 보고서가 없으면 평가하지 않음(judge 0회) |
| report 1회 예외 | `completed`, `… report → retry:report → evaluate → pass` | 빈 보고서 평가가 사라짐 |
| Judge 항상 예외 | `unverified`, `evaluate → retry:quality_evaluator → end:unverified` | |
| Judge 4항목 상시 미달 | `unverified`, 진입 17 | |
| 판정 한쪽 고정 | `completed_with_gaps`, 진입 15 | |
| 4관점 부족 + 금지 표현 | `unverified`, 진입 11 | |
| 종합 항상 예외 | `completed_with_gaps`, 진입 6 | |

### 3.3 병적 단계 상한 (Engineer가 "이론적 최악 30회 이상"이라고 인정한 경우)
`patho3.py`: 평가 Judge가 coverage 미달 대상을 tech → market → stakeholder → domain 순서로 돌려 지목하게 해서 관점 재조사를 1회씩 따로 소진시켰다.
- `MAX_STEPS=60`(상한이 사실상 없는 경우): 자연 종료까지 **진입 41회 / superstep 82**. 재조사 8회 × (재조사·종합·보고서·평가) + 보고서 재작성 2회 → `end:unverified`. 이론적 최악은 실재하고, 그 크기는 41회로 측정됐다.
- **기본값 `MAX_STEPS=20`**: 진입 21회, superstep 42에서 `end:step_limit`으로 정상 종료한다. 직전 보고서는 평가까지 마쳤고(`… report → evaluate → fail:coverage → end:step_limit`), `recursion_limit`은 60이다. `MAX_STEPS=8`과 `40`도 정상 종료했다.
- 다른 고장 조합(`patho.py`: 4관점 1회 부족 + 종합 1회 예외 + 매 종합 4관점 추가 근거 요청 + Judge 3항목 상시 미달)을 `max_steps` 1~15에 대해 모두 돌렸다. **12개 모두** 보고서가 있고, 마지막 보고서를 평가한 뒤 `end:*`로 끝났다.
- 예외: `build_graph(policy=Policy(max_steps=30))`에 기본 `run_config()`를 함께 쓰면 `GraphRecursionError`가 난다. `recursion_limit`이 Policy가 아니라 config의 `MAX_STEPS`에서 계산되기 때문이다(D-20). app 경로는 둘 다 config를 쓰므로 영향이 없다.

### 3.4 재작업 전달
- `cache.py`(반복 1 재현): RAG 41회 중 피드백 포함 14회, domain 재검색이 실제로 일어났다. **해결.**
- `short.py`(신규): market이 기술마다 출처를 1개만 인용하면 Supervisor가 `dispatch:market`을 결정한다(결정적 충분성 검사가 동작함). 그런데 재검색 웹 질의가 `"DeepSeek-V2 MLA market/mla 고유 출처 1개(<2)"`, `"ITME CXL hybrid memory market/mla 고유 출처 1개(<2)"`처럼 **부족 사유 문장 그대로**이고, 다른 기술 이름과도 섞여 있다(D-16). 재조사에서도 출처를 1개만 인용했는데 이전 시도의 출처와 합쳐 2개가 되어 "충분"으로 통과했다(D-17).

### 3.5 재개 (반복 1의 `resume.py`, evidence 저장소 변경 후)
market(병렬 중) 크래시, quality_evaluator 크래시, report 크래시 모두 해당 노드만 다시 실행하고 `completed`로 끝났다. 결정 로그도 `… report → evaluate → pass → end:passed`로 일관된다. **회귀 없음.**

### 3.6 Send 페이로드 축소의 회귀 여부
`SEND_KEYS = (user_query, trace_id, retry_counts, feedback, step_count)`. 4개 관점 에이전트와 guard가 State에서 읽는 키를 grep으로 모두 찾았다: `retry_counts`, `feedback`, `trace_id`. 셋 다 포함돼 있다. **빠진 필드 없음.** LangSmith 컨텍스트 실험에서도 `ContextThreadPoolExecutor`의 `submit`과 `map` 모두 부모 run id가 일치했다.

### 3.7 app 경로 (Fake로 끝까지 실행, `appsim.py`)

| 경로 | 결과 |
|---|---|
| `app.run()` → report 크래시 → `app.run(resume=trace_id)` | `completed`, verified, 4p. 관점과 종합은 다시 실행하지 않고 report만 2회 |
| 종료된 trace를 다시 `--resume` | 노드 실행 없이 다시 내보내기만 함 |
| `--report-only` (새 형식 final_state, 95KB, `excerpt_ref` 75건) | `completed`, report와 judge 각 1회. 보고서 프롬프트 발췌는 900자로 hydrate됨 |
| `--report-only` 직전에 `data/cache` 삭제 | `completed`. 그러나 발췌가 **300자로 조용히 축소**됨(D-18) |
| `--report-only` (이전 형식 State) | `completed_with_gaps`, verified. seed를 offload한 뒤 hydrate해서 발췌 1,006자 |
| `--export-only` (이전 State 복사본) | 9p. `run_logs.json` 보존. exit 2(평가 결과 없음, 의도대로) |
| `--resume x --query q`, `--query q --export-only` | argparse 오류(의도대로) |

`pytest -q`: **41 passed**. `scripts/validate_prompt_package.py`: OK.

## R2-4. 남은 결함과 새 결함

| ID | 심각도 | 루브릭 항목 | 증거 | 재현 방법 | 기대 동작 |
|---|---|---|---|---|---|
| D-01 | **Blocker (미해결)** | 동적 동작 실증, 재현성, Output | `outputs/`에 `Agent_*.md/pdf`, `decisions_*.jsonl`, `outputs/tracing/*.png`가 없다. | `ls outputs/` | `scripts/capture_checklist.md` 순서대로 키가 있는 로컬에서 1회 실행하고 산출물과 캡처를 커밋한다(아래 R2-5의 확인 기준). |
| D-16 | Minor (신규) | 패턴 정합성(재작업 질의) | `policy.py:58-60`의 `_feedback(missing)`은 `rewritten_queries`를 `missing`으로 그대로 채운다. 여기에 `evidence_shortfalls` 사유가 앞에 붙는다(`policy.py:138`). `tools.retry_queries`는 이것을 기술 이름과 결합하면서 다른 기술 이름과도 섞는다. | `short.py`: 웹 질의 `"DeepSeek-V2 MLA market/mla 고유 출처 1개(<2)"`, `"ITME CXL hybrid memory market/mla …"` | 출처 부족 사유는 질의 힌트로 바꾼다(예: `independent sources`). `관점/기술:` 사유는 해당 기술 질의에만 붙인다. D-10과 같은 원칙이다. |
| D-17 | Minor (신규) | 패턴 정합성(충분성 판단), State | `quality.py:79-87`의 `evidence_shortfalls`는 State에 쌓인 모든 시도의 evidence를 센다. 현재 결과가 인용하지 않은 이전 시도의 출처도 포함된다. | `short.py`: 매 시도에 출처를 1개만 인용했는데 두 시도를 합쳐 2개가 되어 `dispatch:market` 1회 뒤 통과 | 충분성은 현재 결과가 인용한 출처(`cited_ids`, 또는 최신 `attempt`의 evidence)로 센다. |
| D-18 | Minor (신규) | 품질 평가 노드(Groundedness) | `evidence_store.hydrate`는 원문이 없으면 축약본(300자)을 아무 표시 없이 그대로 쓴다(`evidence_store.py:69-70`). `data/cache/`는 `.gitignore` 대상이라, 새로 clone한 뒤 `--report-only`를 하면 항상 이 경로를 탄다. | `appsim.py`: cache 삭제 후 report-only 발췌 최대 300자, 평가는 `completed` | 원문이 없는 항목 수를 평가 결과나 `validation.json`에 경고로 남긴다. 또는 `--report-only`에서 저장소가 없으면 실행을 막고 안내한다. |
| D-19 | Minor (D-05 잔여) | State Schema(지속성 비용) | evidence 축약 후에도 이전 실행 규모로 State에 약 228KB가 남는다. SqliteSaver는 체크포인트마다 State 전체를 저장한다(Fake 재작업 실행 21개 = 1.86MB). | `python scripts/measure_state_size.py` | 실제 실행 뒤 체크포인트 크기를 README에 실측치로 적는다. 더 줄이려면 evidence 메타(claim 등)도 줄이고 State에는 id만 둔다. 루브릭 만점의 필수 조건은 아니고 근거 문장과 실측의 일치가 핵심이다. |
| D-20 | Minor (신규) | 재현성 | `observability.run_config`의 기본 `recursion_limit`은 config의 `MAX_STEPS`에서 계산되고 `Policy.max_steps`와 연결되지 않는다. | `patho2.py`: `Policy(max_steps=30)`과 기본 `run_config` → `GraphRecursionError: Recursion limit of 60` | Policy에서 recursion_limit을 계산하거나(`(policy.max_steps+FINALIZE_STEPS)*2+10`), `build_graph`와 `run_config`가 같은 값을 쓰게 한다. |
| P-01 | 참고 (점수 무관) | 프로세스 | Engineer가 PM 소유 문서 `docs/DEV_PLAN.md`의 §3 다이어그램과 규칙 1을 수정했다(커밋 18bca72). 내용은 QA 권고와 일치한다. | `git diff 7ed9f10..HEAD -- docs/DEV_PLAN.md` | PM이 승인하거나 되돌린다. |

## R2-5. 판정과 다음 반복 지시

**판정: 반려(84/100).** 코드는 실제 실행에 들어갈 준비가 됐다. 오프라인으로 검증할 수 있는 항목은 Minor 5건만 남았다. 이 상태에서 실제 실행으로 D-01을 해결하면 97점이고, 아래 1~4까지 끝내면 100점이 가능하다.

1. **D-01(필수)**: `scripts/capture_checklist.md` 순서대로 실제 실행 1회. 다음 반복에서 QA가 확인할 기준은 아래와 같다.
   - `validation.json`의 `trace_id`, `decisions_{trace_id}.jsonl` 파일명, LangSmith run metadata의 `trace_id`, 콘솔 첫 줄이 모두 같다.
   - LangSmith 트리에서 RAG LLM 호출이 노드 run 아래 자식으로 붙어 있다(D-02 수정의 실증).
   - 결정 로그에 `evaluate`가 있고, 최소 1회의 재작업(dispatch 재할당·reinvestigate·rewrite 중 하나)이 사유와 함께 있다. 없으면 재작업이 실제로 일어나지 않은 정상 경로이므로, Fake 시나리오 테스트를 함께 증빙으로 제시한다.
   - `Agent_*.pdf`가 10p 이하이고 SUMMARY와 REFERENCE가 있으며 `validation.json`이 `passed`다. 또는 `completed_with_gaps`나 `unverified`이면 그 사유가 보고서 7장에 드러나 있다.
   - `run_logs.json`과 `final_state.json`이 새 실행물이다.
2. **D-16, D-17**: 재작업 질의 생성과 충분성 계산을 고치고, `short.py`와 같은 시나리오를 테스트로 추가한다.
3. **D-18**: hydrate에 원문이 없을 때 경고나 차단을 남긴다.
4. **D-20, D-19**: recursion_limit을 Policy와 연동하고, 실제 실행 뒤 체크포인트 실측치를 README에 반영한다.

---

# 반복 3

- 검증일 2026-10-07 · 대상 커밋 `19b6a08..359f184` (Engineer 커밋 5개: a1d0f1b, cdec3c1, c6e7e0c, 6e5ea67, 359f184)
- PM 결정에 따라 채점했다. 아래 세 가지는 결함으로 보지 않는다.
  - D-18: 원문이 없으면 명시적 경고(비차단)를 남기고 `--report-only`는 즉시 실패한다 — 승인.
  - D-19: 발췌를 160자로 줄이고 실측치와 잔여 증가를 README에 정직하게 적는 선에서 "지속성 비용"을 충분한 것으로 본다 — 승인.
  - P-01: DEV_PLAN 개정 메모 — 승인.
- 반복 1·2의 재현 스크립트를 모두 다시 실행했다: 엣지 덤프, 고장 주입 7종, 캐시/재작업, 출처 부족, 병적 단계 상한 3종, SQLite 재개, app 경로 시뮬레이션, 원문 저장소 삭제, State 크기 실측. 실제 API 실행과 LangSmith 캡처는 여전히 **미검증**이다.

## R3-1. 총점

**87 / 100. 반려 사유는 D-01 하나뿐이다. 코드는 로컬 실제 실행 준비가 됐다(Yes).**

| 항목 | 배점 | 반복 1 | 반복 2 | 반복 3 | 실행 후 최대 | 근거 |
|---|---|---|---|---|---|---|
| 패턴 적용 정합성 | 20 | 15 | 19 | **20** | 20 | D-16 해결: 재검색 힌트가 기술별 구조(`queries_by_tech`)로 바뀌어 사유 문장이 질의에 들어가지 않는다. D-17 해결: 최신 시도의 `source_units`로 충분성을 센다. 작업 노드 사이 엣지 0개, 단일 conditional 분기 유지. |
| 동적 동작 실증 | 20 | 11 | 14 | **14** | 20 | 코드 쪽 준비는 끝났다(결정 로그 사유, `evaluate` 결정, 스레드 컨텍스트 전파 테스트 통과). 실제 LangSmith 트레이스는 미검증이라 6점을 보류한다. |
| State Schema 설계 | 20 | 17 | 19 | **20** | 20 | D-19는 PM 기준을 충족한다. README `지속성 비용`(`README.md:94-104`)에 300자·160자 실측표와 "SqliteSaver가 superstep마다 State 전체를 다시 저장" 잔여 증가를 명시했다. QA 재측정값과 일치한다(이전 실행 evidence 179,972B, Fake 재작업 체크포인트 21개 1,495,368B). |
| 품질 평가 노드 | 15 | 12 | 14 | **15** | 15 | D-18 해결(승인 기준): 저장소를 지우고 평가하면 `cited_missing_full_text: 42/42`, Judge 프롬프트에 `TRUNCATED` 경고, `eval_result.warnings`가 기록된다. 4항목 판정, 규칙 하드 게이트, 원인별 루프는 유지된다. |
| 코드 구조·모듈 분리 | 5 | 4 | 5 | **5** | 5 | 변동 없음. README 트리와 실제 트리가 일치한다. |
| 실행 결과 재현성 | 10 | 6 | 7 | **7** | 10 | `pytest` **48 passed**. 고장 주입과 단계 상한 시나리오가 모두 정상 종료하고, app 4개 경로가 모두 동작한다. 실제 실행은 미검증이라 3점 보류. |
| Output 보고서 | 10 | 5 | 6 | **6** | 10 | 이전 State를 내보내면 9p다. 새 파이프라인의 `Agent_*` 보고서가 없어 4점 보류. |
| **합계** | **100** | **70** | **84** | **87** | **100** | 보류 13점(동적 6, 재현성 3, Output 4)은 D-01 완료로만 회복된다. |

## R3-2. 반복 2 결함 재검증

| ID | 판정 | 재검증 근거 (QA 직접 실행) |
|---|---|---|
| D-16 | **해결** | `short.py`(market이 기술마다 출처 1개만 인용): 재검색 질의가 `"DeepSeek-V2 MLA additional independent sources analysis"`, `"ITME CXL hybrid memory additional independent sources analysis"`로 바뀌었다. 사유 문장과 다른 기술 이름이 섞이지 않는다. 평가 기반 재조사도 `queries_by_tech` 힌트만 쓴다. |
| D-17 | **해결** | 같은 시나리오에서 이제 `dispatch:market`이 2회 실행되고 한도를 소진하면 gap `market: … 고유 출처 1개(<2, 최신 시도 기준)`가 남고 `completed_with_gaps`로 끝난다. 반복 2에서는 누적 출처로 1회 만에 통과했다. |
| D-18 | **해결(PM 승인 기준)** | 저장소를 지운 뒤 `--report-only`는 `RuntimeError: --report-only 불가: Evidence 75/75건의 원문이 저장소에 없습니다…`로 즉시 실패한다. `--export-only`는 `validation.json`의 `warnings`와 `evidence_store.missing_full_text=75`를 남긴다. 평가 노드는 축약 경고를 Judge 프롬프트와 결과에 기록한다. |
| D-19 | **해결(PM 승인 기준)** | 160자로 줄였다. `measure_state_size.py`를 다시 돌려 README 수치와 일치함을 확인했다. 잔여 증가(체크포인트마다 State 전체 저장)를 README가 숨기지 않고 적었다. |
| D-20 | **해결** | `Policy(max_steps=30)`은 진입 33회에서 `end:step_limit`, `Policy(max_steps=50)`은 41회에서 자연 종료(`end:unverified`)했다. `GraphRecursionError`는 없다. `recursion_limit`은 `build_graph(...).with_config(recursion_limit=policy.recursion_limit)`에서 Policy로 계산한다(단일 출처). 저장소 안에서 삭제된 `config.RECURSION_LIMIT`를 import하는 곳은 0건이다. |
| P-01 | **종결** | `DEV_PLAN.md`에 "2026-10-07 개정… PM 승인" 메모가 있다. |

## R3-3. 회귀 검증 (반복 1·2 스크립트 재실행)

| 대상 | 결과 |
|---|---|
| 엣지 덤프 | 작업 노드 7개가 모두 `supervisor`로만 복귀한다. 분기는 `{'supervisor': ['route']}` 하나. **회귀 없음** |
| 고장 주입 7종 | 반복 2와 같은 경로와 종료 상태다(모두 예외 → `end:report_failed`, Judge 상시 미달 → 진입 17회 `end:unverified` 등). 실패한 보고서는 평가하지 않는다. **회귀 없음** |
| 평가 기반 domain 재조사 | RAG 41회 중 피드백 포함 14회. 이제 지시가 `queries_by_tech`로 기술별로 나뉜다. **회귀 없음** |
| 병적 단계 상한 | 기본값(MAX_STEPS=20)은 진입 21회, superstep 42에서 `end:step_limit`이고 직전 보고서까지 평가했다. 상한이 없으면(60) 자연 최악은 41회다. `patho.py`의 max_steps 1~15 12종은 모두 마지막 보고서를 평가한 뒤 `end:*`로 끝났다. **회귀 없음** |
| SQLite 재개 | market(병렬 중)·quality_evaluator·report 크래시 모두 해당 노드만 다시 실행하고 `completed`로 끝났다. **회귀 없음** |
| 병렬 reducer | `test_parallel_send_writes_are_merged` 통과. `source_units`는 guard가 관점 결과 안에 넣으므로(`supervisor/guard.py:25-29`) 병렬 쓰기 키가 겹치지 않는다. **회귀 없음** |
| 트레이스 컨텍스트 | `test_worker_threads_inherit_graph_run_context` 통과. 스레드풀 코드는 반복 2 이후 바뀌지 않았다. **회귀 없음** |
| app 경로 (Fake 시뮬레이션) | run → report 크래시 → `--resume` → `completed`(report만 재실행). 종료된 trace를 다시 resume하면 내보내기만 한다. `--report-only`는 새 형식 State에서 hydrate 900자, 이전 형식 State에서는 `completed_with_gaps`. 저장소를 지우면 즉시 실패. `--export-only`(이전 State)는 9p이고 `run_logs.json`을 보존한다. **회귀 없음** |

## R3-4. 남은 결함

| ID | 심각도 | 루브릭 항목 | 내용 | 기대 동작 |
|---|---|---|---|---|
| D-01 | **Blocker** | 동적 동작 실증, 재현성, Output | 실제 실행 산출물과 LangSmith 캡처가 없다(코드 결함이 아니라 증빙 미제출). | 아래 R3-5의 종결 기준을 모두 충족한다. |
| D-21 | Minor (신규, 감점 없음) | 패턴(재검색 품질) | 에이전트나 종합이 자유 서술로 보고한 부족 항목은 접두어·기술명만 지운 뒤 질의가 된다. 그래서 `"ITME CXL hybrid memory 상용 고객사 발표 근거 없음"`처럼 부정 표현이 섞인 질의가 나간다. 기술 범위 지정은 올바르다(itme에만 붙음). | 선택 개선: `근거 없음`, `미기재`, `미충족` 같은 부정·메타 표현을 질의에서 제거한다. 루브릭 만점 판정에는 영향이 없다. |

## R3-5. 판정

**코드는 로컬 실제 실행과 캡처를 할 준비가 됐다: Yes.** 오프라인으로 검증할 수 있는 루브릭 항목(패턴 20, State 20, 품질 15, 구조 5)은 만점이다. 남은 13점은 D-01을 실행으로 증명해야만 회복된다. 실행이 아래 기준을 모두 통과하면 다음 반복에서 **100점**으로 판정할 수 있다.

### D-01 종결 기준 (팀이 로컬 실행 후 확인하고 커밋할 것)

1. **실행 전**: 로컬에서 `python -m pytest -q`가 48개 통과하는지 확인한다. 한글 TrueType 폰트를 설치한다(없으면 `validation.json`에 "PDF 조판 검사 불가" 이슈가 남는다). `.env`에 `LANGSMITH_TRACING=true`와 3개 키를 넣는다.
2. **실행**: `python app.py 2>&1 | tee outputs/run_console.log`. 종료 코드가 0이어야 한다. 2이면 `validation.json`의 `issues`가 보고서 7장 한계점에 드러나 있어야 한다(`completed_with_gaps`나 `unverified` 사유 명시). 1이면 실패이므로 다시 실행한다.
3. **상관(trace_id) 일치**: 다음 4곳의 값이 모두 같아야 한다.
   - 콘솔 첫 줄 `trace_id=…`
   - `outputs/decisions_{trace_id}.jsonl`의 파일명과 각 행의 `trace_id`
   - `validation.json`의 `trace_id`
   - LangSmith root run(`kv-eval-supervisor`, tag `pattern:supervisor`)의 metadata `trace_id`
4. **LangSmith 트리 캡처** (`outputs/tracing/*.png`):
   - root 아래에 `supervisor`와 작업 노드가 번갈아 나오는 화면
   - **tech 또는 domain 노드 안에 RAG LLM 호출이 자식 run으로 붙은 화면**(D-02 수정의 실증. root 밖에 고아 run이 따로 생기면 안 된다)
   - `quality_evaluator` 노드의 4항목 결과 화면
   - 캡처에 API 키와 개인정보가 보이지 않아야 한다.
5. **결정 로그 내용**:
   - `decisions_*.jsonl`(그리고 같은 내용의 `run_logs.json`)에 `dispatch:…`, `synthesis`, `report`, `evaluate`, `end:*` 결정이 각각 `reason`과 함께 있다.
   - 재작업(`dispatch:<관점>` 재할당, `reinvestigate:`, `rewrite:report` 중 하나)이 1회 이상이면 그 사유와 `retry_counts`가 `validation.json`과 일치한다.
   - 재작업이 0회인 정상 경로였다면, 동적 동작 증빙은 Fake 시나리오 테스트(`tests/test_scenarios.py`)로 대신한다고 README에 명시한다.
6. **보고서**:
   - `outputs/Agent_판교_9반_김민정_김태동_임동건_김동욱_이재겸_박규리.pdf`와 `.md`
   - `validation.json`의 `pdf_pages`가 10 이하(목표 9 이하)
   - `## SUMMARY`와 `## REFERENCE`가 각 1회
   - 본문 인용이 모두 REFERENCE에 있다(`validation.json`의 `issues`에 인용·목차 이슈가 없다)
   - 보고서 내용은 결정 로그와 일치해야 한다: gaps가 있으면 7장 "근거 공백 (Supervisor 기록)"에 같은 항목이 나온다.
7. **산출물 교체와 커밋**:
   - `outputs/final_state.json`, `validation.json`, `run_logs.json`, `decisions_{trace_id}.jsonl`, `Agent_*.md/pdf`, `tracing/*.png`, `run_console.log`를 커밋한다.
   - `checkpoints.sqlite`와 `data/cache/`는 커밋하지 않는다(gitignore 대상).
   - `validation.json`의 `warnings`에 원문 저장소 누락 경고가 **없어야** 한다(같은 머신에서 실행했다면 0건).
8. **README 갱신**:
   - "실행 예시"를 실제 실행의 결정 경로로 바꾼다(또는 추가한다).
   - `지속성 비용` 표에 실제 실행의 최종 State와 체크포인트 누적 크기를 추가한다. 반복 2에서 README가 약속한 갱신이다.
   - LangSmith 캡처 이미지를 연결한다.
9. **(선택) 재개 시연**: 실행 중 Ctrl+C → `python app.py --resume <trace_id>`. 콘솔에서 완료된 관점이 다시 실행되지 않는 것을 확인하고 캡처한다.

다음 반복에서 QA는 위 1~8을 커밋된 파일끼리 대조한다(trace_id, 결정 로그, validation, 보고서 gaps, 캡처). 모두 맞으면 만점으로 판정한다.
