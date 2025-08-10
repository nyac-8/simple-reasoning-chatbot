"""Writer node - formats final responses from reasoning"""

from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger
from ..state import State
from ..prompts import WRITER_PROMPT
from ..utils import compose_messages_to_prompt, extract_current_question
from ..llm import CustomLLM


def writer_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    Writer node that synthesizes reasoning into final answer.
    
    Pure function that:
    - READS: messages (with all reasoning)
    - RETURNS: {"messages": [...final_answer], "final_answer": str}
    """
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    logger.info(f"Writer started for thread {thread_id}")
    
    # Initialize LLM
    llm = CustomLLM(temperature=0.0)
    
    # Get state
    messages = state.get("messages", [])
    history = state.get("history", [])
    
    # Extract question
    current_question = extract_current_question(messages)
    
    # Build prompt with reasoning steps
    prompt_parts = [WRITER_PROMPT]
    
    # Add reasoning steps
    reasoning_messages = [
        msg for msg in messages 
        if isinstance(msg, AIMessage) and msg.additional_kwargs.get("type") == "reasoning"
    ]
    
    if reasoning_messages:
        prompt_parts.append("\n=== REASONING PROCESS ===")
        for i, msg in enumerate(reasoning_messages, 1):
            prompt_parts.append(f"Step {i}: {msg.content}")
    
    # Add task instruction
    prompt_parts.append("""\n=== YOUR TASK ===
Synthesize the above reasoning into a clear, well-structured response that directly answers the user's question.
Be concise but thorough. Use the reasoning to support your answer.""")
    
    # Compose full prompt
    system_prompt = "\n".join(prompt_parts)
    prompt = compose_messages_to_prompt(
        messages=[m for m in messages if isinstance(m, HumanMessage)],
        system_prompt=system_prompt,
        history=history
    )
    
    logger.debug(f"Synthesizing {len(reasoning_messages)} reasoning steps")
    
    # Generate final answer
    final_answer = llm.generate_content(prompt)
    
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