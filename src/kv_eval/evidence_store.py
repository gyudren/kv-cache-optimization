"""Evidence 발췌 원문 저장소: `data/cache/{trace_id}/evidence_{agent}.json` (State에는 짧은 발췌와 참조만).

근거 원문(논문 청크 최대 약 1,500자, 웹 본문)을 State에 그대로 두면 체크포인트마다 전체가 다시
직렬화된다. 작업 노드가 끝날 때(guard) 원문을 디스크에 쓰고, State의 evidence에는
`excerpt`(앞 STATE_EXCERPT_CHARS자)와 `excerpt_ref`만 남긴다. 원문이 필요한 곳(보고서의 인용 카탈로그,
품질 평가 Judge의 인용 발췌)은 hydrate()로 원문을 되살려 쓴다.
"""
from __future__ import annotations
import json
import threading
from hashlib import sha256
from pathlib import Path
from .config import STATE_EXCERPT_CHARS
from .rag.cache import cache_dir

_LOCK = threading.Lock()  # 같은 프로세스의 병렬 Send가 같은 trace 폴더에 쓸 때 직렬화


def _key(ev: dict) -> str:
    raw = f"{ev.get('source_id', '')}|{ev.get('claim', '')}|{ev.get('page') or ev.get('url') or ''}"
    return sha256(raw.encode()).hexdigest()[:16]


def _path(trace_id: str, agent: str) -> Path:
    return cache_dir(trace_id) / f"evidence_{agent}.json"


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def offload(trace_id: str, agent: str, evidence: list[dict]) -> list[dict]:
    """원문을 디스크에 쓰고 State용 축약 evidence를 돌려준다."""
    if not evidence:
        return []
    path = _path(trace_id, agent)
    compact, store = [], {}
    for ev in evidence:
        excerpt = ev.get("excerpt") or ""
        if ev.get("excerpt_ref") or len(excerpt) <= STATE_EXCERPT_CHARS:
            compact.append(ev)
            continue
        key = _key(ev)
        store[key] = excerpt
        compact.append({**ev, "excerpt": excerpt[:STATE_EXCERPT_CHARS], "excerpt_ref": f"{trace_id}/{agent}/{key}"})
    if store:
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            merged = {**_read(path), **store}
            path.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    return compact


def hydrate(evidence: list[dict]) -> list[dict]:
    """excerpt_ref가 있는 항목의 발췌를 원문으로 되살린다(원문이 없으면 축약본 유지)."""
    files: dict[tuple[str, str], dict] = {}
    out = []
    for ev in evidence:
        ref = ev.get("excerpt_ref")
        if not ref:
            out.append(ev)
            continue
        trace_id, agent, key = ref.split("/", 2)
        if (trace_id, agent) not in files:
            files[(trace_id, agent)] = _read(_path(trace_id, agent))
        full = files[(trace_id, agent)].get(key)
        out.append({**ev, "excerpt": full} if full else ev)
    return out


def source_unit(ev: dict) -> str:
    """고유 출처 단위: 웹은 URL, 논문은 (문서, 페이지)."""
    if ev.get("source_type") == "web":
        return ev.get("url", "")
    return f"{ev.get('doc_id')}:{ev.get('page')}"


def source_units(evidence: list[dict]) -> dict[str, list[str]]:
    """이번 시도에서 기술별로 수집·인용한 고유 출처(Supervisor 충분성 검사 입력)."""
    out: dict[str, set[str]] = {}
    for ev in evidence:
        out.setdefault(ev.get("technology", ""), set()).add(source_unit(ev))
    return {tech: sorted(units) for tech, units in out.items()}
