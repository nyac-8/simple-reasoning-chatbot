"""Main LangGraph definition for the reasoning chatbot"""

from langgraph.graph import StateGraph, END
from loguru import logger
from .state import State
from .agents.orchestrator import orchestrator_node
from .agents.writer import writer_node


def should_continue(state: State) -> str:
    """
    Conditional edge function to determine next node.
    Returns "writer" if ready to answer, "orchestrator" to continue reasoning.
    """
    ready = state.get("ready_to_answer", False)
    
    logger.debug(f"Evaluating continuation: ready={ready}")
    
    if ready:
        logger.info("Routing to writer node")
        return "writer"
    else:
        logger.info("Continuing orchestrator reasoning loop")
        return "orchestrator"


def create_graph():
    """
    Creates and compiles the LangGraph workflow.
    """
    logger.info("Creating reasoning chatbot graph")
    
    # Create the graph
    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("writer", writer_node)
    
    # Set entry point
    graph.set_entry_point("orchestrator")
    
    # Add conditional edge from orchestrator
    graph.add_conditional_edges(
        "orchestrator",
        should_continue,
        {
            "orchestrator": "orchestrator",  # Loop back for more reasoning
            "writer": "writer"  # Hand off to writer
        }
    )
    
    # Writer always goes to END
    graph.add_edge("writer", END)
    
    # Compile the graph
    workflow = graph.compile()
    
    logger.info("Graph created and compiled successfully")
    
    return workflow