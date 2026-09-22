"""Synthesis consumes existing perspectives and deduplicated evidence: NO search."""
from __future__ import annotations
from typing import Any
from ..prompts import prompt_template
from ..state import deduplicate_evidence
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
        + repr({k: state.get(k, {}) for k in ("tech_result", "market_result", "stakeholder_result", "domain_result")})
        + "\nEvidence source IDs: " + repr(src)
        + "\nRevision feedback: " + repr(state.get("review_feedback", {}).get("synthesis", {}))
        + "\nPrevious synthesis: " + repr(state.get("synthesis_result", {})), SynthesisAssessment,
    )
    outcome = assessment.model_dump()
    for tech in ("mla", "itme"):
        if len(outcome["conflicts"].get(tech, [])) < 2:
            outcome["evidence_gaps"].append(f"{tech}: 근거를 갖춘 상충 사례 최소 2건 미충족")
    return {"synthesis_result": outcome, "logs": [{"node": "synthesis", "attempt": state["retry_counts"]["synthesis"],
                                                  "result": "complete", "gate": not bool(outcome["evidence_gaps"])}]}
