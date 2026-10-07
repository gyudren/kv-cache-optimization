"""종합 Agent: 신규 검색 없이 기존 관점 결과와 중복 제거된 Evidence만 쓴다."""
from __future__ import annotations
from typing import Any
from ..evaluation.quality import gap_lines
from ..prompts import prompt_template
from ..config import PERSPECTIVES
from ..state import deduplicate_evidence, perspective, prompt_view
from ..schemas import SynthesisAssessment


def synthesis_node(state: dict, llm: Any) -> dict:
    evidence = deduplicate_evidence(state["evidence"])
    # 인용 매핑에 필요한 필드와 claim만 넘긴다.
    src = [{k: ev.get(k) for k in ("source_id", "claim", "doc_id", "page", "url", "technology")}
           for ev in evidence]
    assessment = llm.generate_structured(
        prompt_template("synthesis") + "\n" + "Synthesize four views (technical TRL, market, stakeholder, domain). Use ONLY supplied evaluated results and verified citations. "
        "Present agreements and at least two EVIDENCED conflicts per technology if possible; if not, list gap rather than fabricate. "
        "Describe trade-offs neutrally, no winner/recommendation. Put a perspective in needs_source_agents only when a specific missing source blocks one of its verdicts, and name that gap in evidence_gaps. "
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
