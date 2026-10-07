"""Synthesis consumes existing perspectives and deduplicated evidence: NO search."""
from __future__ import annotations
from typing import Any
from ..evaluation.quality import gap_lines
from ..prompts import prompt_template
from ..config import PERSPECTIVES
from ..state import deduplicate_evidence, perspective, prompt_view
from ..schemas import SynthesisAssessment


def synthesis_node(state: dict, llm: Any) -> dict:
    evidence = deduplicate_evidence(state["evidence"])
    # Keep citation mapping and summaries, avoid inventing research.
    src = [{k: ev.get(k) for k in ("source_id", "claim", "doc_id", "page", "url", "technology")}
           for ev in evidence]
    assessment = llm.generate_structured(
        prompt_template("synthesis") + "\n" + "Synthesize four views (technical TRL, market, stakeholder, domain). Use ONLY supplied evaluated results and verified citations. "
        "Present agreements and at least two EVIDENCED conflicts per technology if possible; if not, list gap rather than fabricate. "
        "Describe trade-offs neutrally, no winner/recommendation. If sources must be revisited, return names in needs_source_agents. "
        "Set needs_revision for synthesis-only expression problems.\n"
        + repr({k: prompt_view(perspective(state, k)) for k in PERSPECTIVES})
        + "\nEvidence source IDs: " + repr(src)
        + "\nKnown evidence gaps recorded by the supervisor (keep them as gaps): " + repr(gap_lines(state.get("gaps", [])))
        + "\nRevision feedback: " + repr(state.get("feedback", {}).get("synthesis", {}))
        + "\nPrevious synthesis: " + repr(state.get("synthesis", {})), SynthesisAssessment,
    )
    outcome = assessment.model_dump()
    for tech in ("mla", "itme"):
        if len(outcome["conflicts"].get(tech, [])) < 2:
            outcome["evidence_gaps"].append(f"{tech}: 근거를 갖춘 상충 사례 최소 2건 미충족")
    outcome["attempt"] = state["retry_counts"]["synthesis"]
    return {"synthesis": outcome}
