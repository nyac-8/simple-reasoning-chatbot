"""Orchestrator agent - handles reasoning loops and decision making"""

from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from ..state import State
from ..prompts import ORCHESTRATOR_PROMPT
from ..utils import format_conversation_history

# Configuration constants
MAX_REASONING_STEPS = 20
ORCHESTRATOR_TEMPERATURE = 0.0




def orchestrator_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    Orchestrator agent that performs reasoning loops.
    Decides when reasoning is sufficient and ready to hand off to writer.
    
    Follows LangGraph node signature: (state, config) -> state_updates
    """
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    logger.info(f"Orchestrator started for thread {thread_id}")
    
    # Initialize Gemini model with structured output
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=ORCHESTRATOR_TEMPERATURE
    )
    
    # Create JSON parser - simple, no Pydantic
    parser = JsonOutputParser()
    
    # Get current state
    history = state.get("history", [])
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
    
    # Build structured prompt with clear sections
    system_content = ORCHESTRATOR_PROMPT
    
    # Add conversation history if exists
    if history:
        history_section = format_conversation_history(history)
        system_content += f"\n\n{history_section}"
    
    # Build messages list
    messages = [SystemMessage(content=system_content)]
    
    # Add the current question
    messages.append(HumanMessage(content=f"Question: {current_question}"))
    
    # Add previous reasoning steps if any
    if reasoning_steps:
        messages.append(SystemMessage(content="\n=== YOUR REASONING SO FAR ==="))
        for step in reasoning_steps:
            messages.append(SystemMessage(content=f"- {step.content}"))
    
    # Add task instruction with format instructions
    task_instruction = """\n=== YOUR TASK ===
Provide your next reasoning step. Think step-by-step about the question.
Decide if you have enough reasoning to provide a final answer.

Respond in JSON format:
{
    "thinking": "Your reasoning process here",
    "ready_to_answer": true or false
}"""
    
    messages.append(SystemMessage(content=task_instruction))
    
    # Get reasoning step
    logger.debug(f"Generating reasoning step {reasoning_count + 1}")
    logger.info(f"Calling LLM for orchestrator reasoning step {reasoning_count + 1}")
    
    try:
        # Create chain with parser
        chain = model | parser
        response = chain.invoke(messages)
        
        # Extract thinking and decision from parsed response
        # JsonOutputParser returns a dict
        reasoning_content = response.get("thinking", "")
        ready_to_answer = response.get("ready_to_answer", False)
        
        # Ensure at least 2 reasoning steps before allowing ready_to_answer
        if ready_to_answer and reasoning_count < 1:  # 0-indexed, so < 1 means less than 2 steps
            logger.debug("Overriding ready_to_answer=True since we need at least 2 reasoning steps")
            ready_to_answer = False
        
        # Create reasoning step with the thinking content
        reasoning_step = AIMessage(
            content=reasoning_content,
            additional_kwargs={"type": "reasoning"}
        )
        new_reasoning_steps = reasoning_steps + [reasoning_step]
        
        logger.debug(f"Reasoning step {reasoning_count + 1}: {reasoning_content[:100]}...")
        
    except Exception as e:
        logger.error(f"Failed to parse orchestrator response: {e}")
        # Fallback: if we've done at least 2 steps, mark as ready
        ready_to_answer = reasoning_count >= 2
        new_reasoning_steps = reasoning_steps
    
    logger.info(f"Reasoning step {reasoning_count + 1} complete. Ready to answer: {ready_to_answer}")
    
    # Return state updates
    return {
        "reasoning_steps": new_reasoning_steps,
        "reasoning_count": reasoning_count + 1,
        "ready_to_answer": ready_to_answer
    }