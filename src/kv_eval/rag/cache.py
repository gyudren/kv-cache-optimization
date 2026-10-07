"""RAG 답변 캐시: `data/cache/{trace_id}/rag_{agent}.json` (State에는 키만 둔다).

재시도 시 근거가 충분했던 질문은 이전 답변을 재사용한다(answer_with_cache). 이 캐시는 근거 원문
전체를 담고 있어 State에 넣으면 체크포인트마다 수백 KB가 복제되므로 디스크에 두고,
State의 `cache_keys[agent]`에는 파일 경로만 기록한다. trace_id 단위로 분리되어
`--resume`하면 같은 캐시를 다시 읽는다.
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path


def cache_dir(trace_id: str) -> Path:
    # trace_id는 uuid4지만 경로 조작을 막기 위해 안전한 문자만 남긴다.
    safe = re.sub(r"[^A-Za-z0-9_-]", "", trace_id) or "default"
    return Path(os.getenv("RAG_CACHE_DIR", "data/cache")) / safe


def cache_path(trace_id: str, agent: str) -> Path:
    return cache_dir(trace_id) / f"rag_{agent}.json"


def load(trace_id: str, agent: str) -> dict:
    path = cache_path(trace_id, agent)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}  # 손상된 캐시는 버리고 다시 검색한다(정확성 > 비용)


def save(trace_id: str, agent: str, data: dict) -> str:
    path = cache_path(trace_id, agent)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)
