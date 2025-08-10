"""Main entry point for the reasoning chatbot with tool support"""

import os
import sys
from typing import List, Tuple
from uuid import uuid4
from loguru import logger
from langchain_core.messages import HumanMessage, AIMessage
from src.graph import create_graph
from src.tools import REPLTool, get_tavily_tool
from dotenv import load_dotenv


def initialize_tools():
    """Initialize available tools for the chatbot."""
    tools = []
    
    # Always add REPL tool
    tools.append(REPLTool())
    logger.info("Initialized REPL tool")
    
    # Add Tavily if API key is available
    if os.getenv("TAVILY_API_KEY"):
        try:
            tools.append(get_tavily_tool())
            logger.info("Initialized Tavily search tool")
        except Exception as e:
            logger.warning(f"Could not initialize Tavily tool: {e}")
    else:
        logger.warning("TAVILY_API_KEY not found - search tool disabled")
    
    return tools


def run_conversation():
    """Run an interactive conversation with the chatbot."""
    # Load environment variables
    load_dotenv()
    
    # Initialize tools
    tools = initialize_tools()
    
    # Create the graph
    workflow = create_graph()
    
    # Session management
    session_id = str(uuid4())
    history: List[Tuple[str, str]] = []
    
    print("\n=== Reasoning Chatbot v2 (with Tools) ===")
    print("Available tools:", [tool.name for tool in tools])
    print("Type 'quit' to exit, 'clear' to reset history\n")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            history = []
            print("History cleared.")
            continue
        
        if not user_input:
            continue
        
        # Create thread ID for this Q&A
        thread_id = str(uuid4())
        
        # Initialize state for this question
        initial_state = {
            "session_id": session_id,
            "thread_id": thread_id,
            "messages": [HumanMessage(content=user_input)],
            "history": history,
            "context": {},
            "tools": tools,
            "ready_to_answer": False,
            "final_answer": None
        }
        
        try:
            # Run the workflow
            logger.info(f"Processing question: {user_input[:50]}...")
            result = workflow.invoke(
                initial_state,
                {"configurable": {"thread_id": thread_id}}
            )
            
            # Extract final answer
            final_answer = None
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.metadata.get("type") == "final_answer":
                    final_answer = msg.content
                    break
            
            if final_answer:
                print(f"\nAssistant: {final_answer}")
                # Add to history
                history.append((user_input, final_answer))
            else:
                print("\nAssistant: I couldn't generate a response. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            print(f"\nError: {e}")


def main():
    """Main entry point."""
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    
    try:
        run_conversation()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()