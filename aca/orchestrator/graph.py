"""LangGraph state machine for the six-agent pipeline."""
# See docs/02-workflow.md for the full state diagram.
# from langgraph.graph import StateGraph
# from aca.orchestrator.state import WorkflowState
# from aca.agents import analyzer, keyword, brief, writer, reviewer, publisher
#
# def build_graph():
#     g = StateGraph(WorkflowState)
#     g.add_node("analyzer",  analyzer.run)
#     g.add_node("keyword",   keyword.run)
#     g.add_node("brief",     brief.run)
#     g.add_node("writer",    writer.run)
#     g.add_node("reviewer",  reviewer.run)
#     g.add_node("publisher", publisher.run)
#
#     g.set_entry_point("analyzer")
#     g.add_edge("analyzer", "keyword")
#     g.add_edge("keyword",  "brief")
#     g.add_edge("brief",    "writer")
#     g.add_edge("writer",   "reviewer")
#     g.add_conditional_edges(
#         "reviewer",
#         lambda s: "writer" if s.verdict.decision == "revise" and s.revise_count < 2 else "publisher",
#     )
#     return g.compile()
