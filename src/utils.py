"""Utility functions for message handling and prompt construction."""

import sys
from typing import List, Tuple, Optional
from langchain_core.messages import (
    BaseMessage, 
    HumanMessage, 
    AIMessage, 
    SystemMessage,
    ToolMessage,
    get_buffer_string
)
from loguru import logger

# Configure loguru - only once
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)


def messages_to_prompt(messages: List[BaseMessage]) -> str:
    """
    Convert a list of messages to a single prompt string using LangChain's official method.
    
    This uses LangChain's get_buffer_string which formats messages as:
    - Human: <content>
    - AI: <content>
    - System: <content>
    
    Args:
        messages: List of BaseMessage objects
        
    Returns:
        A formatted prompt string
    """
    logger.debug(f"Converting {len(messages)} messages to prompt")
    
    # Use LangChain's official message-to-string converter
    prompt = get_buffer_string(messages)
    
    logger.debug(f"Generated prompt of length {len(prompt)}")
    
    return prompt


def get_logger(name: str):
    """Get a logger instance with the given name."""
    return logger.bind(name=name)


def format_conversation_history(history: List[Tuple[str, str]]) -> str:
    """
    Format conversation history into a clear, structured string.
    
    Args:
        history: List of (question, answer) tuples
        
    Returns:
        Formatted history string
    """
    if not history:
        return ""
    
    history_parts = ["=== PREVIOUS CONVERSATION ==="]
    
    for i, (question, answer) in enumerate(history, 1):
        history_parts.append(f"\n[Turn {i}]")
        history_parts.append(f"Human: {question}")
        history_parts.append(f"Assistant: {answer}")
    
    return "\n".join(history_parts)


def compose_messages_to_prompt(
    messages: List[BaseMessage], 
    system_prompt: Optional[str] = None,
    history: Optional[List[Tuple[str, str]]] = None
) -> str:
    """
    Compose messages and optional context into a single prompt string.
    
    Args:
        messages: List of BaseMessage objects from state
        system_prompt: Optional system prompt to prepend
        history: Optional conversation history
        
    Returns:
        Formatted prompt string
    """
    prompt_parts = []
    
    # Add system prompt if provided
    if system_prompt:
        prompt_parts.append(f"<system>\n{system_prompt}\n</system>")
    
    # Add history if provided
    if history:
        history_str = format_conversation_history(history)
        prompt_parts.append(f"\n{history_str}")
    
    # Process messages
    for msg in messages:
        if isinstance(msg, HumanMessage):
            prompt_parts.append(f"\n<human>\n{msg.content}\n</human>")
        elif isinstance(msg, AIMessage):
            # Check for metadata to determine type
            msg_type = msg.metadata.get("type", "assistant") if hasattr(msg, "metadata") else "assistant"
            if msg_type == "reasoning":
                prompt_parts.append(f"\n<thinking>\n{msg.content}\n</thinking>")
            else:
                prompt_parts.append(f"\n<assistant>\n{msg.content}\n</assistant>")
        elif isinstance(msg, ToolMessage):
            prompt_parts.append(f"\n<tool>\n{msg.content}\n</tool>")
        elif isinstance(msg, SystemMessage):
            # Skip system messages in state (shouldn't be there per our design)
            logger.warning("Found SystemMessage in state messages - skipping")
            continue
    
    return "\n".join(prompt_parts)


def extract_current_question(messages: List[BaseMessage]) -> str:
    """
    Extract the current question from messages (first HumanMessage).
    
    Args:
        messages: List of BaseMessage objects
        
    Returns:
        The current question string
    """
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def count_reasoning_steps(messages: List[BaseMessage]) -> int:
    """
    Count reasoning steps in messages.
    
    Args:
        messages: List of BaseMessage objects
        
    Returns:
        Number of reasoning steps
    """
    count = 0
    for msg in messages:
        if isinstance(msg, AIMessage):
            # Check metadata for type
            if hasattr(msg, "metadata") and msg.metadata.get("type") == "reasoning":
                count += 1
            # Also check additional_kwargs for backwards compatibility
            elif hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get("type") == "reasoning":
                count += 1
    return count