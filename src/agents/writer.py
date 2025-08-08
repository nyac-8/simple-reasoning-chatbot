"""Writer agent - formats final responses from reasoning"""

from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from ..state import State
from ..prompts import WRITER_PROMPT
from ..utils import get_logger

# Configuration constants
WRITER_TEMPERATURE = 0.0

logger = get_logger("writer")


def writer_agent(state: State) -> Dict[str, Any]:
    """
    Writer agent that takes reasoning steps and creates a polished final answer.
    """
    logger.info(f"Writer started for thread {state.get('thread_id', 'unknown')}")
    
    # Initialize Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=WRITER_TEMPERATURE  # Lower temperature for more consistent formatting
    )
    
    # Get state components
    current_question = state.get("current_question", "")
    reasoning_steps = state.get("reasoning_steps", [])
    messages = state.get("messages", [])
    history = state.get("history", [])
    
    # Build the writing prompt
    writing_messages = [
        SystemMessage(content=WRITER_PROMPT)
    ]
    
    # Add conversation context if exists
    if messages:
        writing_messages.append(
            SystemMessage(content="Conversation context:")
        )
        # Only include human/AI exchanges, not system messages
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage)) and not msg.additional_kwargs.get("type") == "reasoning":
                writing_messages.append(msg)
    
    # Add the current question
    writing_messages.append(
        HumanMessage(content=current_question)
    )
    
    # Add reasoning steps
    if reasoning_steps:
        writing_messages.append(
            SystemMessage(content="Reasoning process to synthesize:")
        )
        for step in reasoning_steps:
            writing_messages.append(
                SystemMessage(content=f"Reasoning: {step.content}")
            )
    
    # Request final answer
    writing_messages.append(
        SystemMessage(content="Now, synthesize this reasoning into a clear, well-structured response to the user's question:")
    )
    
    # Generate final answer
    logger.debug("Generating final answer from reasoning")
    response = model.invoke(writing_messages)
    
    final_answer = response.content
    
    # Update history with Q&A pair
    new_history = history + [(current_question, final_answer)]
    
    logger.info("Final answer generated successfully")
    
    # Return state updates
    return {
        "final_answer": final_answer,
        "history": new_history,
        "messages": messages + [
            HumanMessage(content=current_question),
            AIMessage(content=final_answer, additional_kwargs={"type": "final_answer"})
        ]
    }