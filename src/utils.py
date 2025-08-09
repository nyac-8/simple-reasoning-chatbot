"""Utility functions for message handling and prompt construction."""

import sys
from typing import List, Tuple, Optional
from langchain_core.messages import BaseMessage, get_buffer_string
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