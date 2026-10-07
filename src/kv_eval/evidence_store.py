"""근거 원문 저장소: `data/cache/{trace_id}/evidence_{agent}.json`.

체크포인트가 커지지 않도록 State에는 앞부분 발췌와 excerpt_ref만 두고, 원문은 hydrate()로 되살린다.
"""
from __future__ import annotations
import json
import threading
from hashlib import sha256
from pathlib import Path
from .config import STATE_EXCERPT_CHARS
from .rag.cache import cache_dir

_LOCK = threading.Lock()  # 병렬 Send가 같은 trace 폴더에 쓰는 것을 직렬화


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
        if len(excerpt) <= STATE_EXCERPT_CHARS:
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


def hydrate(evidence: list[dict], stats: dict | None = None) -> list[dict]:
    """excerpt_ref가 있는 항목의 발췌를 원문으로 되살린다.

    원문이 없으면 축약본을 두고 excerpt_truncated를 표시하며, stats에 개수를 센다.
    """
    files: dict[tuple[str, str], dict] = {}
    out, missing = [], 0
    for ev in evidence:
        ref = ev.get("excerpt_ref")
        if not ref:
            out.append(ev)
            continue
        trace_id, agent, key = ref.split("/", 2)
        if (trace_id, agent) not in files:
            files[(trace_id, agent)] = _read(_path(trace_id, agent))
        full = files[(trace_id, agent)].get(key)
        if full:
            out.append({**ev, "excerpt": full})
        else:
            missing += 1
            out.append({**ev, "excerpt_truncated": True})
    if stats is not None:
        stats["missing_full_text"] = stats.get("missing_full_text", 0) + missing
    return out


def missing_full_text(evidence: list[dict]) -> int:
    stats: dict = {}
    hydrate(evidence, stats)
    return stats["missing_full_text"]


def source_unit(ev: dict) -> str:
    """고유 출처 단위: 웹은 URL, 논문은 doc_id. 같은 논문의 다른 페이지는 같은 출처다."""
    if ev.get("source_type") == "web":
        return ev.get("url", "")
    return f"doc:{ev.get('doc_id')}"


def source_units(evidence: list[dict]) -> dict[str, list[str]]:
    """이번 시도의 기술별 고유 출처. Supervisor 충분성 검사에 쓴다."""
    out: dict[str, set[str]] = {}
    for ev in evidence:
        out.setdefault(ev.get("technology", ""), set()).add(source_unit(ev))
    return {tech: sorted(units) for tech, units in out.items()}
