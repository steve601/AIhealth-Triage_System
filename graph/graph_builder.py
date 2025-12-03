from agentNodes.intakeNode import intake
from agentNodes.explainNode import explain
from agentNodes.ragNode import get_relevant_evidence
from agentNodes.triageNode import conduct_triage
from models.agentState import AgentState
from langgraph.graph import START,END,StateGraph

def buid_graph():
    graph = StateGraph(AgentState)

    graph.add_node('intake',intake)
    graph.add_node('RAG',get_relevant_evidence)
    graph.add_node('triage',conduct_triage)
    graph.add_node('explain',explain)

    graph.add_edge(START,'intake')
    graph.add_edge('intake','RAG')
    graph.add_edge('RAG','triage')
    graph.add_edge('triage','explain')
    graph.add_edge('explain',END)

    return graph.compile()