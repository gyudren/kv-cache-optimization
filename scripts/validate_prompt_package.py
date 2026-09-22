#!/usr/bin/env python3
"""Validate the prompt package without third-party dependencies."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "prompts"

PROMPT_FILES = [
    "00_common_contract.md",
    "01_master_agent.md",
    "02_technical_research_agent.md",
    "03_market_evaluation_agent.md",
    "04_stakeholder_evaluation_agent.md",
    "05_domain_evaluation_agent.md",
    "06_synthesis_agent.md",
    "07_report_agent.md",
    "08_result_validator.md",
]

AGENT_FILES = PROMPT_FILES[1:]

ALLOWED_PLACEHOLDERS = {
    "user_query",
    "technology",
    "attempt",
    "review_feedback",
    "rag_context",
    "web_results",
    "validated_evidence",
    "perspective_results",
    "as_of_date",
    "state_json",
    "synthesis_result",
    "target_agent",
    "retry_limit",
    "agent_result",
    "evidence",
}

REQUIRED_CRITERIA = {
    "02_technical_research_agent.md": [f"Q{i}" for i in range(1, 8)],
    "03_market_evaluation_agent.md": [f"M{i}" for i in range(1, 4)],
    "04_stakeholder_evaluation_agent.md": [f"S{i}" for i in range(1, 4)],
    "05_domain_evaluation_agent.md": [f"D{i}" for i in range(1, 8)],
}


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    errors: list[str] = []

    for name in PROMPT_FILES:
        path = PROMPT_DIR / name
        if not path.is_file():
            fail(errors, f"missing prompt file: {path.relative_to(ROOT)}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    prompt_texts = {
        name: (PROMPT_DIR / name).read_text(encoding="utf-8")
        for name in PROMPT_FILES
    }

    for name in AGENT_FILES:
        text = prompt_texts[name]
        if "## SYSTEM PROMPT" not in text:
            fail(errors, f"{name}: missing SYSTEM PROMPT section")
        if "## TASK TEMPLATE" not in text:
            fail(errors, f"{name}: missing TASK TEMPLATE section")

    placeholder_pattern = re.compile(r"\{\{([a-z_]+)\}\}")
    used_placeholders: set[str] = set()
    for name, text in prompt_texts.items():
        for placeholder in placeholder_pattern.findall(text):
            used_placeholders.add(placeholder)
            if placeholder not in ALLOWED_PLACEHOLDERS:
                fail(errors, f"{name}: unknown placeholder {{{{{placeholder}}}}}")

    readme_text = (PROMPT_DIR / "README.md").read_text(encoding="utf-8")
    for placeholder in sorted(used_placeholders):
        if f"`{{{{{placeholder}}}}}`" not in readme_text:
            fail(errors, f"README.md: undocumented placeholder {{{{{placeholder}}}}}")

    for name, criteria in REQUIRED_CRITERIA.items():
        text = prompt_texts[name]
        for criterion in criteria:
            if not re.search(rf"\b{re.escape(criterion)}\b", text):
                fail(errors, f"{name}: missing criterion {criterion}")

    schema_path = ROOT / "schemas" / "evidence.schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        fail(errors, f"invalid evidence schema: {exc}")
    else:
        required = set(schema.get("required", []))
        for key in {"source_id", "agent", "technology", "claim", "source_type"}:
            if key not in required:
                fail(errors, f"evidence schema: required key missing: {key}")

    project_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if 'MARKET["시장 평가 에이전트<br/>RAG"]' in project_readme:
        fail(errors, "README.md: market Agent is incorrectly labelled as RAG")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"OK: {len(PROMPT_FILES)} prompt files validated")
    print(f"OK: {len(used_placeholders)} documented placeholders validated")
    print("OK: Q1-Q7, M1-M3, S1-S3, D1-D7 criteria present")
    print("OK: evidence schema is valid JSON with core required keys")
    return 0


if __name__ == "__main__":
    sys.exit(main())

