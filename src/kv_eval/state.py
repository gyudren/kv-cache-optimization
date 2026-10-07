"""Supervisor 패턴 State 계약 (DEV_PLAN §5).

필드는 두 묶음으로 나눈다.
- 제어(control): Supervisor가 라우팅 판단에 읽는 값. 결과 본문 구조와 무관하다.
- 페이로드(payload): 하위 에이전트가 만든 결과 본문. Supervisor는 `sufficient`·`missing` 외에는 읽지 않는다.

`Send`로 여러 관점 에이전트가 같은 superstep에 동시에 쓰므로, 동시에 쓰일 수 있는 필드는 모두
reducer로 병합 규칙을 정한다(reducer가 없으면 InvalidUpdateError 또는 마지막 값만 남는다).
결정 로그 본문과 RAG 캐시는 State 밖(외부 JSONL·디스크)에 두고 State에는 직전 결정과 캐시 키만 둔다.
"""
from __future__ import annotations
import re
from typing import Annotated, Any, Literal, TypedDict
from .config import FOLLOWUP_LIMITS, PERSPECTIVES, RETRY_LIMITS, STATE_EXCERPT_CHARS

NodeStatus = Literal["pending", "running", "done", "failed", "skipped"]  # skipped = 실패 후 한도 소진(공백 기록)
# 관점별 근거 충분도. excluded = 재시도 상한을 넘겨 근거 공백으로 기록하고 제외한 관점
PerspectiveStatus = Literal["pending", "sufficient", "insufficient", "failed", "excluded"]


# ---- reducers ---------------------------------------------------------------------------------
def merge_dict(left: dict | None, right: dict | None) -> dict:
    """키 단위 병합. 병렬 Send가 서로 다른 키(관점)에 쓰면 둘 다 남고, 같은 키면 나중 값이 이긴다."""
    return {**(left or {}), **(right or {})}


def evidence_key(ev: dict) -> tuple[str, str, object]:
    return (ev.get("source_id", ""), ev.get("claim", ""), ev.get("page") or ev.get("url"))


def merge_evidence(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """에이전트 단위 교체 + dedup-append. 발췌는 STATE_EXCERPT_CHARS로 제한한다.

    에이전트가 다시 실행되면 그 에이전트의 이번 결과가 전체 근거다(충분했던 RAG 답은 캐시에서 다시 실려 온다).
    이전 시도 근거를 남겨 두면 보고서·편향 검사가 지금은 쓰지 않는 근거까지 세므로, 새 쓰기에 `agent`가 있는
    근거가 들어오면 같은 agent의 기존 근거를 먼저 지운다. agent가 없는 입력(seed·테스트)은 그대로 이어 붙인다.
    원문은 작업 노드 경계(guard)에서 evidence_store로 먼저 옮겨지고 excerpt_ref가 붙는다. 여기의 자르기는
    그 경로를 거치지 않은 입력(초기 seed 등)까지 State 크기를 보장하는 마지막 상한이다.
    """
    replaced = {ev.get("agent") for ev in right or []} - {None}
    out = [ev for ev in left or [] if ev.get("agent") not in replaced]
    seen = {evidence_key(ev) for ev in out}
    for ev in right or []:
        key = evidence_key(ev)
        if key in seen:
            continue
        seen.add(key)
        excerpt = ev.get("excerpt") or ""
        out.append({**ev, "excerpt": excerpt[:STATE_EXCERPT_CHARS]} if len(excerpt) > STATE_EXCERPT_CHARS else ev)
    return out


def merge_unique(left: list | None, right: list | None) -> list:
    return list(dict.fromkeys([*(left or []), *(right or [])]))


GapKind = Literal["insufficient", "agent_failed", "step_limit", "followup_exhausted", "evaluation", "node_failed"]


class Gap(TypedDict):
    """Supervisor가 더 조사하지 않기로 하고 남긴 근거 공백. 보고서 7장과 편향·커버리지 규칙이 읽는다."""
    perspective: str          # tech/market/stakeholder/domain, 또는 synthesis/report/quality_evaluator
    technology: str | None    # mla/itme. 관점 전체에 해당하면 None
    kind: GapKind
    criterion: str | None     # kind == "evaluation"일 때 미달 항목
    detail: str               # 확인하지 못한 내용(보고서에 그대로 쓴다)


def make_gap(perspective: str, kind: GapKind, detail: str, technology: str | None = None,
             criterion: str | None = None) -> Gap:
    return {"perspective": perspective, "technology": technology, "kind": kind, "criterion": criterion,
            "detail": " ".join(str(detail).split())}


def split_scope(text: str) -> tuple[str | None, str]:
    """에이전트 결함 문장의 '관점/기술: 내용' 접두어에서 기술을 떼어 낸다."""
    match = re.match(r"^\w+/(mla|itme):\s*(.*)$", str(text), re.S)
    return (match.group(1), match.group(2)) if match else (None, re.sub(r"^\w+:\s*", "", str(text)))


def merge_gaps(left: list[Gap] | None, right: list[Gap] | None) -> list[Gap]:
    out = list(left or [])
    for gap in right or []:
        if gap not in out:
            out.append(gap)
    return out


class PerspectiveResult(TypedDict, total=False):
    """관점 에이전트 결과. Supervisor는 앞의 제어 필드만 읽고, 나머지는 보고서가 쓴다."""
    sufficient: bool                     # 필수 결함이 없는가(Supervisor 판단 입력)
    missing: list[str]                   # 필수 결함 → 재조사 사유
    missing_optional: list[str]          # 확인 못 한 세부 항목 → 한계점
    llm_sufficient: bool                 # LLM 자기 판단(기록용)
    attempt: int
    source_units: dict[str, list[str]]   # 이번 시도의 기술별 고유 출처(guard가 기록)
    summary: str
    trl: dict[str, str]                  # tech
    trl_basis: dict[str, str]            # tech
    details: dict[str, Any]              # tech
    technologies: dict[str, dict]        # market / stakeholder
    items: list[dict]                    # domain


class GraphState(TypedDict, total=False):
    user_query: str
    # 제어: Supervisor가 라우팅에 쓰는 값
    trace_id: str                       # = LangGraph thread_id = LangSmith metadata = 결정 로그 파일명
    step_count: int                     # Supervisor 진입 횟수
    next_agents: list[str]              # 직전 결정. 라우터는 이것만 읽는다
    perspective_status: Annotated[dict[str, str], merge_dict]
    retry_counts: Annotated[dict[str, int], merge_dict]      # 충분성 재조사·실행 실패 재시도
    followup_counts: Annotated[dict[str, int], merge_dict]   # 종합·평가가 요청한 후속 재조사
    node_status: Annotated[dict[str, str], merge_dict]
    last_error: Annotated[dict[str, str], merge_dict]
    feedback: Annotated[dict[str, dict], merge_dict]         # Supervisor → 에이전트 재작업 지시
    eval_result: dict[str, Any]
    last_decision: dict[str, Any]                            # 결정 로그 본문은 외부 JSONL
    gaps: Annotated[list[Gap], merge_gaps]
    status: Literal["running", "completed", "completed_with_gaps", "unverified"]
    # 페이로드: 하위 에이전트 결과
    perspectives: Annotated[dict[str, PerspectiveResult], merge_dict]
    synthesis: dict[str, Any]
    report: str
    evidence: Annotated[list[dict], merge_evidence]          # 발췌는 160자, 원문은 디스크
    cache_keys: Annotated[dict[str, str], merge_dict]


def initial_state(query: str, trace_id: str, seed: dict | None = None) -> GraphState:
    """새 실행의 State. seed가 있으면 이전 실행의 페이로드를 이어받는다(--report-only)."""
    state: GraphState = {
        "user_query": query, "trace_id": trace_id, "step_count": 0, "next_agents": [],
        "perspective_status": {name: "pending" for name in PERSPECTIVES},
        "retry_counts": {name: 0 for name in RETRY_LIMITS},
        "followup_counts": {name: 0 for name in FOLLOWUP_LIMITS},
        "node_status": {name: "pending" for name in (*PERSPECTIVES, "synthesis", "report", "quality_evaluator")},
        "last_error": {}, "feedback": {}, "eval_result": {}, "last_decision": {}, "gaps": [],
        "status": "running",
        "perspectives": {}, "synthesis": {}, "report": "", "evidence": [], "cache_keys": {},
    }
    state.update(seed or {})
    return state


def perspective(state: dict, name: str) -> dict:
    return (state.get("perspectives") or {}).get(name) or {}


def attempt_of(state: dict, name: str) -> int:
    """관점 에이전트의 시도 번호(충분성 재조사 + 후속 재조사)."""
    return int((state.get("retry_counts") or {}).get(name, 0)) + int((state.get("followup_counts") or {}).get(name, 0))


def legacy_gaps(gaps: list) -> list[Gap]:
    """문자열 공백("관점: 내용")을 저장한 이전 final_state.json을 읽을 때 쓴다."""
    out = []
    for gap in gaps or []:
        if isinstance(gap, dict):
            out.append(gap)
            continue
        name = re.match(r"^(\w+)", str(gap))
        technology, detail = split_scope(gap)
        out.append(make_gap(name.group(1) if name else "기타", "insufficient", detail, technology))
    return out


def deduplicate_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, object]] = set()
    out = []
    for ev in items:
        key = evidence_key(ev)
        if key not in seen:
            seen.add(key)
            out.append(ev)
    return out


def prompt_view(result: dict) -> dict:
    """LLM 프롬프트에 넣을 결과 사본. 재시도용 내부 캐시(rag_cache)가 섞여 있으면 제외한다."""
    return {k: v for k, v in (result or {}).items() if k != "rag_cache"}
