"""Load the common contract and role system prompts from the project root.

Supervisor는 LLM을 쓰지 않아 프롬프트가 없다.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPT_FILES = {
    "technology": "02_technical_research_agent.md",
    "market": "03_market_evaluation_agent.md",
    "stakeholder": "04_stakeholder_evaluation_agent.md",
    "domain": "05_domain_evaluation_agent.md",
    "synthesis": "06_synthesis_agent.md",
    "report": "07_report_agent.md",
    "quality_evaluator": "09_quality_evaluator.md",
}
ALLOWED = frozenset(PROMPT_FILES)


def _system_prompt(name: str) -> str:
    path = ROOT / "prompts" / PROMPT_FILES[name]
    text = path.read_text(encoding="utf-8")
    marker = "## SYSTEM PROMPT"
    task_marker = "## TASK TEMPLATE"
    if marker not in text or task_marker not in text:
        raise ValueError(f"Invalid prompt contract sections: {path.name}")
    return text.split(marker, 1)[1].split(task_marker, 1)[0].strip()


def prompt_template(name: str, *additional_names: str) -> str:
    """Combine the common contract with one or more role system prompts.

    TASK TEMPLATE sections are skipped; agents pass task data and output schemas at runtime.
    """
    names = (name, *additional_names)
    unsupported = [item for item in names if item not in ALLOWED]
    if unsupported:
        raise ValueError(f"Unsupported design prompt: {', '.join(unsupported)}")
    common = (ROOT / "prompts" / "00_common_contract.md").read_text(encoding="utf-8").strip()
    return "\n\n".join([common, *(_system_prompt(item) for item in names)])
