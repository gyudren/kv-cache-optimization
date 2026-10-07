# 실제 실행 · LangSmith 캡처 체크리스트

API 키가 있는 로컬에서 아래 순서를 따른다. 실행 1회는 약 15분 걸린다.

## 0. 준비

```bash
pip install -e ".[dev]"
python -m pytest -q                      # 오프라인 테스트가 먼저 통과해야 한다
cp .env.example .env                     # 아래 값 입력
```

`.env` 필수 값: `OPENAI_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT=kv-cache-supervisor`
한글 PDF 폰트가 있어야 한다(macOS 나눔고딕 또는 리눅스 `fonts-nanum`). 없으면 `validation.json`에 "PDF 조판 검사 불가" 이슈가 남는다.

## 1. 실행

```bash
python app.py 2>&1 | tee outputs/run_console.log
```

- 첫 줄 `[trace] trace_id=<UUID> · LangSmith ON (project=kv-cache-supervisor)`의 **trace_id를 기록**한다. OFF로 나오면 키 설정을 확인하고 다시 실행한다.
- 진행 중 `supervisor#N → dispatch:... / evaluate / rewrite:report / reinvestigate:...` 줄이 결정 로그(`outputs/decisions_<trace_id>.jsonl`)와 같은지 확인한다.
- 재개 시연(선택): 실행 중 Ctrl+C로 중단한 뒤 `python app.py --resume <trace_id>`를 돌리고, 완료된 관점이 다시 실행되지 않는 것을 콘솔로 확인한다.

종료 코드: `0` 품질 평가 통과 / `2` 보고서는 생성됐으나 미검증 / `1` 실행 실패. `2`이면 `outputs/validation.json`의 `issues`와 `quality_eval`을 확인한다.

## 2. LangSmith 캡처 (`outputs/tracing/`에 저장)

LangSmith → Projects → `kv-cache-supervisor` → Runs. 먼저 Runs 목록에 metadata 필터 `trace_id = <1단계에서 기록한 값>`을 걸어 완료된 실행 1건만 보이게 한다. 중단된 실행(`running`)이나 다른 실행이 화면에 섞이면 안 된다. 그 run(`kv-eval-supervisor`)을 열어 아래를 캡처한다.

| 파일 | 화면 | 확인할 것 |
|---|---|---|
| `tracing-1.png` | metadata `trace_id` 필터를 건 Runs 목록 | run 1건만 보임, run_name `kv-eval-supervisor`, 태그 `pattern:supervisor`, 완료 상태, 실행 시간 |
| `tracing-2.png` | run 상세 > Metadata | `trace_id`가 콘솔·결정 로그 파일명과 같음 |
| `tracing-3.png` | run 트리: 첫 `supervisor` 다음에 tech·market·stakeholder·domain이 같은 단계에 할당되어 하나씩 차례로 실행된 부분 | Send fan-out, 각 노드 아래 LLM·웹 호출이 자식 run으로 붙음(고아 root run 없음) |
| `tracing-4.png` | 재작업 또는 평가 구간: `supervisor`(dispatch:market 등) → 해당 에이전트 → `supervisor`(evaluate) → `quality_evaluator` | 부족 관점만 재실행, 보고서 후 Supervisor가 평가를 부름 |
| `tracing-5.png` | `quality_evaluator` 노드의 출력(`eval_result.criteria`) | 4항목 passed·score·reason |

재작업이 한 번도 일어나지 않았으면 `tracing-4.png`는 `report → supervisor(evaluate) → quality_evaluator → supervisor(end:passed)` 구간으로 대신한다.

## 3. 커밋할 파일

```bash
git add outputs/Agent_판교_9반_김민정_김태동_임동건_김동욱_이재겸_박규리.md \
        outputs/Agent_판교_9반_김민정_김태동_임동건_김동욱_이재겸_박규리.pdf \
        outputs/validation.json outputs/run_logs.json outputs/final_state.json outputs/corpus_stats.json \
        outputs/decisions_<trace_id>.jsonl outputs/run_console.log outputs/tracing/
```

커밋하지 않는 것: `outputs/checkpoints.sqlite*`, `data/cache/`(둘 다 .gitignore), `.env`.
건드리지 않는 것: `outputs/RAG-Output_*`(이전 과제 제출물), `outputs/legacy_rag/`(이전 구조 실행물).

## 4. 대조 확인

- [ ] `validation.json`의 `trace_id` = `decisions_<trace_id>.jsonl` 파일명 = LangSmith metadata `trace_id`
- [ ] `run_logs.json`의 결정 순서가 LangSmith 트리의 노드 순서와 같음
- [ ] `validation.json`의 `pdf_pages` ≤ 10(목표 9), `passed`, `quality_eval` 4항목
- [ ] 보고서 7장에 `근거 공백 (Supervisor 기록)` 절이 있으면 `validation.json`의 `gaps`와 같음
- [ ] README "실행 예시"를 실제 결정 경로로 갱신
