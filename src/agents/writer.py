"""Writer node - formats final responses from reasoning and tool results"""

from typing import Dict, Any
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger
from ..state import State
from ..prompts import WRITER_PROMPT
from ..llm import CustomLLM


def writer_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    Writer node that synthesizes all messages into a final answer.
    
    Pure function that:
    - READS: messages (complete temporal history including reasoning and tools)
    - RETURNS: {"messages": [...final_answer], "final_answer": str}
    - Uses ALL context to answer the human's question
    """
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    logger.info(f"Writer started for thread {thread_id}")
    
    # Initialize LLM
    llm = CustomLLM(temperature=0.0)
    
    # Get state
    messages = state.get("messages", [])
    history = state.get("history", [])
    
    # Create system prompt for writer
    system_prompt = f"""{WRITER_PROMPT}

=== YOUR TASK ===
The messages below show the complete conversation in temporal order.
This includes:
- The human's question
- AI reasoning steps
- Tool calls and their results
- All context gathered

Based on ALL this information, provide a clear, accurate, and helpful answer to the human's question.
Use the tool results and reasoning to support your response.
DO NOT say you don't have information if tools have provided it.

The messages are in temporal order (oldest first).
Write your response directly to answer the human's original question."""
    
    logger.debug(f"Synthesizing from {len(messages)} messages")
    
    # Pass ALL messages to the LLM - let it see everything
    full_messages = [SystemMessage(content=system_prompt)] + messages
    
    # Generate final answer using all context
    final_answer = llm.generate_from_messages(full_messages)
    
    logger.info("Final answer generated successfully")
    
    # Create final answer message with metadata
    final_message = AIMessage(
        content=final_answer,
        additional_kwargs={"type": "final_answer"},
        metadata={"type": "final_answer", "source": "writer"}
    )
    
    # Return only changed fields
    return {
        "messages": messages + [final_message],
        "final_answer": final_answer
    }