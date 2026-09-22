"""Load the six fixed prompt contracts from the project root (no new Agent)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOWED = {"technology", "market", "stakeholder", "domain", "synthesis", "report"}


def prompt_template(name: str) -> str:
    if name not in ALLOWED:
        raise ValueError(f"Unsupported design prompt: {name}")
    path = ROOT / "prompts" / f"{name}.md"
    return path.read_text(encoding="utf-8")
