"""One Master role, six task Agents; Send fan-out joins selected workers only."""
from __future__ import annotations
from functools import partial
from typing import Any
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from .agents import master, technology, market, stakeholder, domain, synthesis, report
from .state import GraphState


def build_graph(rag: Any, web: Any, llm: Any):
    graph = StateGraph(GraphState)
    graph.add_node("master_init", master.master_init_node)
    graph.add_node("technology", partial(technology.technology_node, rag=rag, llm=llm, web=web))
    graph.add_node("master_tech_gate", master.master_tech_gate_node)
    graph.add_node("master_query_rewrite", master.master_query_rewrite_node)
    graph.add_node("master_dispatch", master.master_dispatch_node)
    graph.add_node("market", partial(market.market_node, web=web, llm=llm))
    graph.add_node("stakeholder", partial(stakeholder.stakeholder_node, web=web, llm=llm))
    graph.add_node("domain", partial(domain.domain_node, rag=rag, llm=llm))
    graph.add_node("master_join", master.master_join_node)
    graph.add_node("master_retry", master.master_retry_node)
    graph.add_node("synthesis", partial(synthesis.synthesis_node, llm=llm))
    graph.add_node("master_synthesis_gate", master.master_synthesis_gate_node)
    graph.add_node("report", partial(report.report_node, llm=llm))
    graph.add_node("master_report_gate", partial(master.master_report_gate_node, llm=llm))

    graph.add_edge(START, "master_init")
    graph.add_edge("master_init", "technology")
    graph.add_edge("technology", "master_tech_gate")
    # SUP_TECH -- 부족(최대 2회) --> QUERY_REWRITE --> TECH / -- 충분 --> SUP_FANOUT
    graph.add_conditional_edges("master_tech_gate", master.route_tech,
                                {"master_query_rewrite": "master_query_rewrite",
                                 "master_dispatch": "master_dispatch"})
    graph.add_edge("master_query_rewrite", "technology")

    def dispatch_routes(state: GraphState) -> list[Send]:
        selected = state["next_agents"]
        if not selected or not set(selected) <= {"market", "stakeholder", "domain"}:
            raise ValueError("Invalid selected perspective fan-out")
        # Each selected Agent writes only its own result plus reducer lists.
        return [Send(name, state) for name in selected]

    # 도달 가능한 노드를 명시해야 그래프 구조가 설계 D-2의 관점별 병렬 평가와 일치한다.
    graph.add_conditional_edges("master_dispatch", dispatch_routes,
                                ["market", "stakeholder", "domain"])
    for name in ("market", "stakeholder", "domain"):
        graph.add_edge(name, "master_join")
    # RESULT_GATE -- 미완료·근거 부족 --> RETRY --> 부족한 Agent만 재할당 / -- 완료 --> SYNTHESIS
    graph.add_conditional_edges("master_join", master.route_join,
                                {"master_retry": "master_retry", "synthesis": "synthesis"})
    graph.add_edge("master_retry", "master_dispatch")
    graph.add_edge("synthesis", "master_synthesis_gate")
    # SUP_SYNTHESIS -- 부족(최대 1회) --> SYNTHESIS / -- 충분 --> REPORT (설계 D-2에 없는 경로는 두지 않는다)
    graph.add_conditional_edges("master_synthesis_gate", master.route_synthesis,
                                {"synthesis": "synthesis", "report": "report"})
    graph.add_edge("report", "master_report_gate")
    graph.add_conditional_edges("master_report_gate", master.route_report,
                                {"report": "report", "end": END})
    return graph.compile()
