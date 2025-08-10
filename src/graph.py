"""Main LangGraph definition for the reasoning chatbot with tool support"""

from langgraph.graph import StateGraph, END
from loguru import logger
from .state import State
from .agents.orchestrator import orchestrator_node
from .agents.writer import writer_node
from .agents.tool_executor import tool_executor_node
from langchain_core.messages import AIMessage


def should_continue(state: State) -> str:
    """
    Conditional edge function to determine next node based on orchestrator's decision.
    
    Checks the metadata of the last reasoning message to determine:
    - "tools" if use_tools=true
    - "writer" if ready_for_final_answer=true
    - "orchestrator" to continue reasoning
    """
    messages = state.get("messages", [])
    ready = state.get("ready_to_answer", False)
    
    # Find the last reasoning message to check its metadata
    last_reasoning = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.metadata.get("type") == "reasoning":
            last_reasoning = msg
            break
    
    # Check metadata for routing decision
    if last_reasoning:
        use_tools = last_reasoning.metadata.get("use_tools", False)
        ready_for_final = last_reasoning.metadata.get("ready_for_final_answer", False)
        
        logger.debug(f"Routing decision: use_tools={use_tools}, ready={ready_for_final}")
        
        if use_tools:
            logger.info("Routing to tool executor")
            return "tools"
        elif ready_for_final or ready:
            logger.info("Routing to writer node")
            return "writer"
    
    # Default: continue reasoning
    logger.info("Continuing orchestrator reasoning loop")
    return "orchestrator"


def create_graph():
    """
    Creates and compiles the LangGraph workflow with tool support.
    
    Graph flow:
    - START -> orchestrator
    - orchestrator -> [orchestrator | tools | writer] based on decision
    - tools -> orchestrator (always back for post-tool reasoning)
    - writer -> END
    """
    logger.info("Creating reasoning chatbot graph with tool support")
    
    # Create the graph
    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("tools", tool_executor_node)
    graph.add_node("writer", writer_node)
    
    # Set entry point
    graph.set_entry_point("orchestrator")
    
    # Add conditional edge from orchestrator
    graph.add_conditional_edges(
        "orchestrator",
        should_continue,
        {
            "orchestrator": "orchestrator",  # Continue reasoning
            "tools": "tools",                # Execute tools
            "writer": "writer"                # Generate final answer
        }
    )
    
    # Tools always return to orchestrator for post-tool reasoning
    graph.add_edge("tools", "orchestrator")
    
    # Writer always goes to END
    graph.add_edge("writer", END)
    
    # Compile the graph
    workflow = graph.compile()
    
    logger.info("Graph created and compiled successfully")
    
    return workflow