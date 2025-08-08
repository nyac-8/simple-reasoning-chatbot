"""Orchestrator agent - handles reasoning loops and decision making"""

import json
from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from ..state import State
from ..prompts import ORCHESTRATOR_PROMPT
from ..utils import get_logger

# Configuration constants
MAX_REASONING_STEPS = 20
ORCHESTRATOR_TEMPERATURE = 0.0
MIN_REASONING_LENGTH = 10

logger = get_logger("orchestrator")


def parse_json_response(response_content: str) -> Dict[str, Any]:
    """Parse JSON response from the model, handling markdown code blocks."""
    try:
        # Handle markdown code blocks
        if response_content.startswith("```json"):
            response_content = response_content.replace("```json", "").replace("```", "").strip()
        elif response_content.startswith("```"):
            response_content = response_content.replace("```", "").strip()
        
        return json.loads(response_content)
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Raw response: {response_content}")
        return {}


def orchestrator_agent(state: State) -> Dict[str, Any]:
    """
    Orchestrator agent that performs reasoning loops.
    Decides when reasoning is sufficient and ready to hand off to writer.
    """
    logger.info(f"Orchestrator started for thread {state.get('thread_id', 'unknown')}")
    
    # Initialize Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=ORCHESTRATOR_TEMPERATURE
    )
    
    # Get current state
    messages = state.get("messages", [])
    reasoning_steps = state.get("reasoning_steps", [])
    reasoning_count = state.get("reasoning_count", 0)
    current_question = state.get("current_question", "")
    
    # Check if we've hit the limit
    if reasoning_count >= MAX_REASONING_STEPS:
        logger.warning(f"Hit max reasoning steps ({MAX_REASONING_STEPS})")
        return {
            "ready_to_answer": True,
            "reasoning_count": reasoning_count
        }
    
    # Build the reasoning prompt
    reasoning_messages = [
        SystemMessage(content=ORCHESTRATOR_PROMPT)
    ]
    
    # Add conversation history if exists
    if messages:
        reasoning_messages.extend(messages)
    
    # Add the current question as a HumanMessage
    reasoning_messages.append(HumanMessage(content=current_question))
    
    # Add previous reasoning steps if any
    if reasoning_steps:
        reasoning_messages.append(
            SystemMessage(content="Your reasoning so far:")
        )
        reasoning_messages.extend(reasoning_steps)
    
    # Request next reasoning step with JSON response
    json_prompt = """Based on the question and any previous reasoning, provide your next reasoning step.

Remember to respond in JSON format:
{
    "thinking": "Your reasoning here",
    "ready_to_answer": true or false
}"""
    
    reasoning_messages.append(SystemMessage(content=json_prompt))
    
    # Get reasoning step
    logger.debug(f"Generating reasoning step {reasoning_count + 1}")
    reasoning_response = model.invoke(reasoning_messages)
    
    response_content = reasoning_response.content.strip()
    
    # Parse JSON response
    response_data = parse_json_response(response_content)
    
    if response_data:
        # Extract thinking and decision from parsed JSON
        reasoning_content = response_data.get("thinking", "")
        ready_to_answer = response_data.get("ready_to_answer", False)
        
        # Validate the response
        if not reasoning_content or len(reasoning_content.strip()) < MIN_REASONING_LENGTH:
            logger.warning(f"Empty or too short reasoning content at step {reasoning_count + 1}")
            # If we've done at least 2 steps, mark as ready
            ready_to_answer = reasoning_count >= 2
            new_reasoning_steps = reasoning_steps
        else:
            # Create reasoning step with the thinking content
            reasoning_step = AIMessage(
                content=reasoning_content,
                additional_kwargs={"type": "reasoning"}
            )
            new_reasoning_steps = reasoning_steps + [reasoning_step]
    else:
        # Fallback: try to use the raw content as reasoning
        if response_content and len(response_content) > MIN_REASONING_LENGTH:
            reasoning_step = AIMessage(
                content=response_content,
                additional_kwargs={"type": "reasoning"}
            )
            new_reasoning_steps = reasoning_steps + [reasoning_step]
        else:
            new_reasoning_steps = reasoning_steps
        
        # Default decision based on step count
        ready_to_answer = reasoning_count >= 2
    
    logger.info(f"Reasoning step {reasoning_count + 1} complete. Ready to answer: {ready_to_answer}")
    
    # Return state updates
    return {
        "reasoning_steps": new_reasoning_steps,
        "reasoning_count": reasoning_count + 1,
        "ready_to_answer": ready_to_answer
    }