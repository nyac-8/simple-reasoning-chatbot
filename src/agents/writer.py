"""Writer agent - formats final responses from reasoning"""

from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from ..state import State
from ..prompts import WRITER_PROMPT
from ..utils import format_conversation_history

# Configuration constants
WRITER_TEMPERATURE = 0.0


def writer_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    Writer agent that takes reasoning steps and creates a polished final answer.
    
    Follows LangGraph node signature: (state, config) -> state_updates
    """
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    logger.info(f"Writer started for thread {thread_id}")
    
    # Initialize Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=WRITER_TEMPERATURE  # Lower temperature for more consistent formatting
    )
    
    # Get state components
    current_question = state.get("current_question", "")
    reasoning_steps = state.get("reasoning_steps", [])
    history = state.get("history", [])
    
    # Build structured prompt with clear sections
    system_content = WRITER_PROMPT
    
    # Add conversation history if exists
    if history:
        history_section = format_conversation_history(history)
        system_content += f"\n\n{history_section}"
    
    # Build messages list
    messages = [SystemMessage(content=system_content)]
    
    # Add the current question
    messages.append(HumanMessage(content=f"Question: {current_question}"))
    
    # Add reasoning steps
    if reasoning_steps:
        messages.append(SystemMessage(content="\n=== REASONING PROCESS ==="))
        for i, step in enumerate(reasoning_steps, 1):
            messages.append(SystemMessage(content=f"Step {i}: {step.content}"))
    
    # Add task instruction
    task_instruction = """\n=== YOUR TASK ===
Synthesize the above reasoning into a clear, well-structured response that directly answers the user's question.
Be concise but thorough. Use the reasoning to support your answer."""
    
    messages.append(SystemMessage(content=task_instruction))
    
    # Generate final answer
    logger.debug("Generating final answer from reasoning")
    logger.info(f"Calling LLM for writer to synthesize {len(reasoning_steps)} reasoning steps")
    
    response = model.invoke(messages)
    final_answer = response.content
    
    # Update history with Q&A pair
    new_history = history + [(current_question, final_answer)]
    
    logger.info("Final answer generated successfully")
    
    # Create updated messages list with just the Q&A
    new_messages = [
        HumanMessage(content=current_question),
        AIMessage(content=final_answer, additional_kwargs={"type": "final_answer"})
    ]
    
    # Return state updates
    return {
        "final_answer": final_answer,
        "history": new_history,
        "messages": new_messages  # Only keep current Q&A in messages
    }