# QA 보고서: 반복 1 (feat/multi-agent-supervisor)

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
