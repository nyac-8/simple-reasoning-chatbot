from typing import TypedDict, List, Dict, Any, Optional, Tuple, Annotated
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool


class State(TypedDict):
    """
    State schema for the reasoning chatbot (v2).
    
    Simplified state with messages as the single source of truth.
    All reasoning steps, tool calls, and responses flow through messages.
    """
    
    session_id: Annotated[str, "Session identifier (managed externally)"]
    thread_id: Annotated[str, "Thread identifier for one Q&A cycle"]
    
    messages: Annotated[List[BaseMessage], "ALL messages in thread (Human, AI, Tool)"]
    history: Annotated[List[Tuple[str, str]], "Past threads (first human + final answer pairs)"]
    
    context: Annotated[Dict[str, Any], "Tool outputs, search results, etc."]
    tools: Annotated[List[BaseTool], "Available tools for v2"]
    
    ready_to_answer: Annotated[bool, "Orchestrator's decision flag"]
    final_answer: Annotated[Optional[str], "Cached final answer for convenience"]