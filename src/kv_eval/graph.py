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
    graph.add_node("technology", partial(technology.technology_node, rag=rag, llm=llm))
    graph.add_node("master_tech_gate", master.master_tech_gate_node)
    graph.add_node("master_dispatch", master.master_dispatch_node)
    graph.add_node("market", partial(market.market_node, web=web, llm=llm))
    graph.add_node("stakeholder", partial(stakeholder.stakeholder_node, web=web, llm=llm))
    graph.add_node("domain", partial(domain.domain_node, rag=rag, llm=llm))
    graph.add_node("master_join", master.master_join_node)
    graph.add_node("synthesis", partial(synthesis.synthesis_node, llm=llm))
    graph.add_node("master_synthesis_gate", master.master_synthesis_gate_node)
    graph.add_node("report", partial(report.report_node, llm=llm))
    graph.add_node("master_report_gate", partial(master.master_report_gate_node, llm=llm))

    graph.add_edge(START, "master_init")
    graph.add_edge("master_init", "technology")
    graph.add_edge("technology", "master_tech_gate")
    graph.add_conditional_edges("master_tech_gate", master.route_tech,
                                {"technology": "technology", "master_dispatch": "master_dispatch"})

    def dispatch_routes(state: GraphState) -> list[Send]:
        selected = state["next_agents"]
        if not selected or not set(selected) <= {"market", "stakeholder", "domain"}:
            raise ValueError("Invalid selected perspective fan-out")
        # Each selected Agent writes only its own result plus reducer lists.
        return [Send(name, state) for name in selected]

    graph.add_conditional_edges("master_dispatch", dispatch_routes)
    for name in ("market", "stakeholder", "domain"):
        graph.add_edge(name, "master_join")
    graph.add_conditional_edges("master_join", master.route_join,
                                {"master_dispatch": "master_dispatch", "synthesis": "synthesis"})
    graph.add_edge("synthesis", "master_synthesis_gate")
    graph.add_conditional_edges("master_synthesis_gate", master.route_synthesis,
                                {"master_dispatch": "master_dispatch", "synthesis": "synthesis", "report": "report"})
    graph.add_edge("report", "master_report_gate")
    graph.add_conditional_edges("master_report_gate", master.route_report,
                                {"report": "report", "end": END})
    return graph.compile()
