"""RAG 답변 캐시: `data/cache/{trace_id}/rag_{agent}.json`.

근거 원문을 담고 있어 State에는 파일 경로(cache_keys)만 둔다. --resume하면 같은 trace_id의 캐시를 다시 읽는다.
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path


def cache_dir(trace_id: str) -> Path:
    # 경로 조작을 막으려고 안전한 문자만 남긴다.
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
        return {}  # 손상된 캐시는 버리고 다시 검색한다


def save(trace_id: str, agent: str, data: dict) -> str:
    path = cache_path(trace_id, agent)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)
