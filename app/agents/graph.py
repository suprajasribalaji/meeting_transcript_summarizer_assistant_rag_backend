from langgraph.graph import StateGraph, END, START
from .state import AgentState
from .nodes import retrieve_context, generate_response


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_response", generate_response)

    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()


transcript_agent = build_graph()