"""그래프 State 정의와 reducer.

Send로 여러 관점이 한 superstep에 동시에 쓰는 필드는 모두 reducer로 병합 규칙을 정해 둔다.
결정 로그 본문과 RAG 캐시는 State 밖에 두고, State에는 직전 결정과 캐시 키만 남긴다.
"""
from __future__ import annotations
import re
from typing import Annotated, Any, Literal, TypedDict
from .config import FOLLOWUP_LIMITS, PERSPECTIVES, REINVESTIGATE_LIMITS, RETRY_LIMITS, STATE_EXCERPT_CHARS

NodeStatus = Literal["pending", "running", "done", "failed", "skipped"]  # skipped: 실패 후 한도를 넘겨 건너뜀
# excluded: 재시도 한도를 넘겨 근거 공백으로 남기고 제외한 관점
PerspectiveStatus = Literal["pending", "sufficient", "insufficient", "failed", "excluded"]


def merge_dict(left: dict | None, right: dict | None) -> dict:
    """키 단위 병합. 같은 키면 나중 값이 이긴다."""
    return {**(left or {}), **(right or {})}


def evidence_key(ev: dict) -> tuple[str, str, object]:
    return (ev.get("source_id", ""), ev.get("claim", ""), ev.get("page") or ev.get("url"))


def merge_evidence(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """에이전트 단위로 교체하며 중복 없이 이어 붙인다.

    재실행한 에이전트는 이번 결과가 전체 근거라 같은 agent의 기존 근거를 지운다. agent가 없는 입력은 그냥 붙인다.
    발췌를 자르는 건 guard를 거치지 않은 입력(seed 등)까지 State 크기를 묶어 두기 위해서다.
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


# one_sided = 반대 방향 근거를 재검색했지만 찾지 못해 한쪽 방향 판정만 남은 관점·기술
GapKind = Literal["insufficient", "one_sided", "agent_failed", "step_limit", "followup_exhausted", "evaluation",
                  "node_failed"]


class Gap(TypedDict):
    """Supervisor가 더 조사하지 않기로 하고 남긴 근거 공백. 보고서 7장과 편향·커버리지 규칙이 읽는다."""
    perspective: str          # 관점 또는 synthesis/report/quality_evaluator
    technology: str | None    # mla/itme, 관점 전체면 None
    kind: GapKind
    criterion: str | None     # evaluation일 때 미달 항목
    detail: str               # 보고서에 그대로 쓴다


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
    """관점 에이전트 결과. Supervisor는 앞쪽 제어 필드만 읽는다."""
    sufficient: bool                     # 필수 결함이 없는가
    missing: list[str]                   # 필수 결함, 재조사 사유가 된다
    missing_optional: list[str]          # 확인 못 한 세부 항목, 한계점으로만 쓴다
    llm_sufficient: bool                 # LLM 자기 판단, 기록용
    attempt: int
    source_units: dict[str, list[str]]   # 이번 시도의 기술별 고유 출처, guard가 채운다
    summary: str
    trl: dict[str, str]                  # tech
    trl_basis: dict[str, str]            # tech
    details: dict[str, Any]              # tech
    technologies: dict[str, dict]        # market / stakeholder
    items: list[dict]                    # domain


class GraphState(TypedDict, total=False):
    user_query: str
    # 제어: Supervisor가 라우팅에 쓰는 값
    trace_id: str                       # thread_id, LangSmith metadata, 결정 로그 파일명에 같이 쓴다
    step_count: int                     # Supervisor 진입 횟수
    next_agents: list[str]              # 라우터가 읽는 직전 결정
    perspective_status: Annotated[dict[str, str], merge_dict]
    retry_counts: Annotated[dict[str, int], merge_dict]      # 충분성 재조사·실행 실패 재시도
    followup_counts: Annotated[dict[str, int], merge_dict]   # 종합이 요청한 후속 재조사
    reinvestigate_counts: Annotated[dict[str, int], merge_dict]  # 품질 평가가 원인으로 지목한 재조사
    node_status: Annotated[dict[str, str], merge_dict]
    last_error: Annotated[dict[str, str], merge_dict]
    feedback: Annotated[dict[str, dict], merge_dict]         # 에이전트에 넘기는 재작업 지시
    eval_result: dict[str, Any]
    last_decision: dict[str, Any]
    gaps: Annotated[list[Gap], merge_gaps]
    status: Literal["running", "completed", "completed_with_gaps", "unverified"]
    # 페이로드: 하위 에이전트 결과
    perspectives: Annotated[dict[str, PerspectiveResult], merge_dict]
    synthesis: dict[str, Any]
    report: str
    evidence: Annotated[list[dict], merge_evidence]          # 축약 발췌만, 원문은 evidence_store
    cache_keys: Annotated[dict[str, str], merge_dict]


def initial_state(query: str, trace_id: str, seed: dict | None = None) -> GraphState:
    """새 실행의 State. seed가 있으면 이전 실행의 페이로드를 이어받는다(--report-only)."""
    state: GraphState = {
        "user_query": query, "trace_id": trace_id, "step_count": 0, "next_agents": [],
        "perspective_status": {name: "pending" for name in PERSPECTIVES},
        "retry_counts": {name: 0 for name in RETRY_LIMITS},
        "followup_counts": {name: 0 for name in FOLLOWUP_LIMITS},
        "reinvestigate_counts": {name: 0 for name in REINVESTIGATE_LIMITS},
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
    """충분성 재조사, 종합 후속 재조사, 평가 재조사를 합친 시도 번호."""
    return sum(int((state.get(field) or {}).get(name, 0))
               for field in ("retry_counts", "followup_counts", "reinvestigate_counts"))


def legacy_gaps(gaps: list) -> list[Gap]:
    """공백을 문자열("관점: 내용")로 저장하던 예전 final_state.json을 읽을 때 쓴다."""
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
    """LLM 프롬프트용 결과 사본. 내부 캐시(rag_cache)는 뺀다."""
    return {k: v for k, v in (result or {}).items() if k != "rag_cache"}
