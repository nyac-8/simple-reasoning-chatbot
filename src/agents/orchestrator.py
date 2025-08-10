"""Orchestrator node - handles reasoning loops and decision making"""

import json
from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger
from ..state import State
from ..prompts import ORCHESTRATOR_PROMPT
from ..utils import (
    compose_messages_to_prompt,
    extract_current_question,
    count_reasoning_steps
)
from ..llm import CustomLLM

# Dynamic configuration
MAX_REASONING_STEPS = 20  # Will be made configurable later


def orchestrator_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    Orchestrator node that performs reasoning loops.
    
    Pure function that:
    - READS: messages, history, context
    - RETURNS: {"messages": [...new_reasoning], "ready_to_answer": bool}
    """
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    logger.info(f"Orchestrator started for thread {thread_id}")
    
    # Initialize LLM
    llm = CustomLLM(temperature=0.0)
    
    # Get state
    messages = state.get("messages", [])
    history = state.get("history", [])
    
    # Extract current question and count reasoning steps
    current_question = extract_current_question(messages)
    reasoning_count = count_reasoning_steps(messages)
    
    # Check if we've hit the limit
    if reasoning_count >= MAX_REASONING_STEPS:
        logger.warning(f"Hit max reasoning steps ({MAX_REASONING_STEPS})")
        return {"ready_to_answer": True}
    
    # Build prompt with reasoning context
    prompt_parts = [ORCHESTRATOR_PROMPT]
    
    # Add previous reasoning if any
    if reasoning_count > 0:
        prompt_parts.append("\n=== YOUR REASONING SO FAR ===")
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.additional_kwargs.get("type") == "reasoning":
                prompt_parts.append(f"- {msg.content}")
    
    # Add task instruction
    prompt_parts.append("""\n=== YOUR TASK ===
Provide your next reasoning step. Think step-by-step about the question.
Decide if you have enough reasoning to provide a final answer.

Respond in JSON format:
{
    "thinking": "Your reasoning process here",
    "ready_to_answer": true or false
}""")
    
    # Compose full prompt
    system_prompt = "\n".join(prompt_parts)
    prompt = compose_messages_to_prompt(
        messages=[m for m in messages if isinstance(m, HumanMessage)],  # Only human messages
        system_prompt=system_prompt,
        history=history
    )
    
    logger.debug(f"Generating reasoning step {reasoning_count + 1}")
    
    try:
        # Define schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "thinking": {"type": "string"},
                "ready_to_answer": {"type": "boolean"}
            },
            "required": ["thinking", "ready_to_answer"]
        }
        
        # Get structured response
        response_json = llm.get_structured_output(prompt, schema)
        response = json.loads(response_json)
        
        reasoning_content = response.get("thinking", "")
        ready_to_answer = response.get("ready_to_answer", False)
        
        # Ensure minimum reasoning steps
        if ready_to_answer and reasoning_count < 1:
            logger.debug("Need at least 2 reasoning steps")
            ready_to_answer = False
        
        # Create new reasoning message with metadata
        new_message = AIMessage(
            content=reasoning_content,
            additional_kwargs={"type": "reasoning"},
            metadata={"type": "reasoning", "source": "orchestrator"}
        )
        
        logger.info(f"Step {reasoning_count + 1} complete. Ready: {ready_to_answer}")
        
        # Return only changed fields
        return {
            "messages": messages + [new_message],
            "ready_to_answer": ready_to_answer
        }
        
    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
        # Fallback
        return {"ready_to_answer": reasoning_count >= 2}