"""Orchestrator node - handles reasoning loops and decision making"""

import json
from typing import Dict, Any, List
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger
from ..state import State
from ..prompts import ORCHESTRATOR_PROMPT
from ..llm import CustomLLM


# Dynamic configuration
MAX_REASONING_STEPS = 30  # Increased for tool usage


def orchestrator_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    Orchestrator node that performs ONE reasoning step and decides next action.
    
    Pure function that:
    - READS: messages (in temporal order), history, tools
    - RETURNS: {"messages": [new_reasoning], "ready_to_answer": bool}
    - Decides: continue thinking, use tools, or ready to answer
    """
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    logger.info(f"Orchestrator started for thread {thread_id}")
    
    # Initialize LLM
    llm = CustomLLM(temperature=0.0)
    
    # Get state
    messages = state.get("messages", [])
    history = state.get("history", [])
    tools = state.get("tools", [])
    
    # Count reasoning steps for safety limit
    reasoning_count = sum(
        1 for m in messages 
        if isinstance(m, AIMessage) and m.metadata.get("type") == "reasoning"
    )
    
    # Check if we've hit the limit
    if reasoning_count >= MAX_REASONING_STEPS:
        logger.warning(f"Hit max reasoning steps ({MAX_REASONING_STEPS})")
        return {"ready_to_answer": True}
    
    # Build prompt - just explain the messages are temporal
    prompt_parts = [ORCHESTRATOR_PROMPT]
    
    # Add tool availability
    if tools:
        prompt_parts.append("\n=== AVAILABLE TOOLS ===")
        for tool in tools:
            prompt_parts.append(f"- {tool.name}: {tool.description}")
    
    # Check if last message was a tool response
    last_was_tool = len(messages) > 0 and isinstance(messages[-1], ToolMessage)
    
    # Task instruction based on context
    if last_was_tool:
        prompt_parts.append("""\n=== YOUR TASK ===
The messages below show the conversation in temporal order (oldest first).
The most recent message is a tool response. Analyze it and decide what to do next.

You MUST acknowledge the tool results in your thinking and reason about next steps.

Respond in JSON format:
{
    "thinking": "Acknowledge tool results and reason about next steps",
    "use_tools": true/false,
    "ready_for_final_answer": true/false
}""")
    else:
        prompt_parts.append("""\n=== YOUR TASK ===
The messages below show the conversation in temporal order (oldest first).
Analyze the current state and decide what to do next.

IMPORTANT: Be efficient! 
- Simple questions may only need ONE reasoning step
- Don't artificially extend reasoning
- Only continue if you genuinely need more analysis

Options:
- Continue thinking: Only if the problem requires deeper analysis
- Use tools: If you need to execute code or search for information  
- Ready for final answer: As soon as you have sufficient reasoning

Respond in JSON format:
{
    "thinking": "Your reasoning process here",
    "use_tools": true/false,
    "ready_for_final_answer": true/false
}

Note: If use_tools is true, be specific in your thinking about what tool is needed.""")
    
    prompt_parts.append("\n=== MESSAGES (in temporal order) ===")
    
    # Create the full prompt
    system_prompt = "\n".join(prompt_parts)
    
    logger.debug(f"Generating reasoning step {reasoning_count + 1}")
    
    try:
        # Define schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "thinking": {"type": "string"},
                "use_tools": {"type": "boolean"},
                "ready_for_final_answer": {"type": "boolean"}
            },
            "required": ["thinking", "use_tools", "ready_for_final_answer"]
        }
        
        # Pass messages directly to LLM - it will handle them temporally
        response_json = llm.get_structured_output_with_messages(
            messages=messages,
            system_prompt=system_prompt,
            schema=schema
        )
        response = json.loads(response_json)
        
        thinking = response.get("thinking", "")
        use_tools = response.get("use_tools", False)
        ready_for_final_answer = response.get("ready_for_final_answer", False)
        
        # Create reasoning message
        reasoning_message = AIMessage(
            content=thinking,
            metadata={
                "type": "reasoning",
                "source": "orchestrator",
                "use_tools": use_tools,
                "ready_for_final_answer": ready_for_final_answer
            }
        )
        
        logger.info(f"Step {reasoning_count + 1}: use_tools={use_tools}, ready={ready_for_final_answer}")
        
        # Return updated messages and ready state
        return {
            "messages": messages + [reasoning_message],
            "ready_to_answer": ready_for_final_answer
        }
        
    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
        # Fallback - let the system decide naturally
        return {"ready_to_answer": reasoning_count > 0}