"""Tool executor node - analyzes reasoning and executes appropriate tools"""

from typing import Dict, Any
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger
from ..state import State
from ..llm import CustomLLM
import json


def tool_executor_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    Tool executor node that analyzes messages and executes appropriate tools.
    
    Adds both the tool call (as AIMessage) and tool response (as ToolMessage) to messages.
    
    Pure function that:
    - READS: messages (in temporal order), tools
    - RETURNS: {"messages": [...with AIMessage(tool_calls) and ToolMessage]}
    - Creates complete audit trail of tool usage
    """
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    logger.info(f"Tool executor started for thread {thread_id}")
    
    # Get state
    messages = state.get("messages", [])
    tools = state.get("tools", [])
    
    if not tools:
        logger.warning("No tools available")
        return {"messages": messages}
    
    # Initialize LLM for tool selection
    llm = CustomLLM(temperature=0.0)
    
    # Build tool descriptions
    tool_descriptions = []
    tool_map = {}
    for tool in tools:
        tool_descriptions.append(f"- {tool.name}: {tool.description}")
        tool_map[tool.name] = tool
    
    # Create prompt for tool selection
    system_prompt = f"""You are a tool executor. Based on the conversation history (in temporal order), 
determine which tool to call and with what arguments.

The most recent AI reasoning message indicates a tool is needed.
Look at the latest reasoning to understand what tool to use.

AVAILABLE TOOLS:
{chr(10).join(tool_descriptions)}

IMPORTANT for python_repl:
- Write complete, executable Python code
- Include print statements to show results
- For calculations, print the final result
- Example: {{"code": "import math\\nresult = math.factorial(8)\\nprint(f'8! = {{result}}')"}}

IMPORTANT for tavily_search:
- Create a clear, specific search query
- Example: {{"query": "current average savings account interest rates USA 2024"}}

The messages below are in temporal order (oldest first).
Based on the most recent reasoning, select and return the appropriate tool call.

Respond in JSON format:
{{
    "tool_name": "python_repl" or "tavily_search",
    "tool_arguments": {{
        "code": "complete python code with print statements" OR
        "query": "specific search query"
    }}
}}"""
    
    try:
        # Define schema for tool selection
        schema = {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "enum": list(tool_map.keys())},
                "tool_arguments": {"type": "object"}
            },
            "required": ["tool_name", "tool_arguments"]
        }
        
        # Pass messages directly to LLM - it sees the full temporal context
        response_json = llm.get_structured_output_with_messages(
            messages=messages,
            system_prompt=system_prompt,
            schema=schema
        )
        response = json.loads(response_json)
        
        tool_name = response.get("tool_name")
        tool_arguments = response.get("tool_arguments", {})
        
        # Handle case where tool_arguments is a JSON string
        if isinstance(tool_arguments, str):
            try:
                tool_arguments = json.loads(tool_arguments)
            except json.JSONDecodeError:
                pass
        
        logger.info(f"Executing tool: {tool_name}")
        
        # Execute the selected tool
        if tool_name in tool_map:
            tool = tool_map[tool_name]
            
            # Prepare the actual arguments for the tool
            if tool_name == "python_repl":
                if isinstance(tool_arguments, dict):
                    code = tool_arguments.get("code", "")
                    actual_args = {"code": code}
                else:
                    code = str(tool_arguments)
                    actual_args = {"code": code}
            elif tool_name == "tavily_search":
                if isinstance(tool_arguments, dict):
                    query = tool_arguments.get("query", "")
                    actual_args = {"query": query}
                else:
                    query = str(tool_arguments)
                    actual_args = {"query": query}
            else:
                actual_args = tool_arguments if isinstance(tool_arguments, dict) else {"input": tool_arguments}
            
            # Create AIMessage with tool call BEFORE execution
            # This shows what the system decided to call
            tool_call_id = f"call_{thread_id}_{len(messages)}"
            tool_call_message = AIMessage(
                content="",  # Tool calls typically have empty content
                tool_calls=[{
                    "id": tool_call_id,
                    "name": tool_name,
                    "args": actual_args
                }],
                metadata={"type": "tool_call", "source": "tool_executor"}
            )
            
            # Execute the tool
            result = tool.invoke(actual_args)
            
            # Create ToolMessage with the response
            tool_message = ToolMessage(
                content=str(result),
                name=tool_name,
                tool_call_id=tool_call_id,  # Links to the tool call
                metadata={"source": "tool_executor", "args": actual_args}
            )
            
            logger.info(f"Tool {tool_name} executed successfully")
            
            # Return both the tool call and the tool response
            return {"messages": messages + [tool_call_message, tool_message]}
        else:
            logger.error(f"Tool {tool_name} not found")
            return {"messages": messages}
            
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        # Add error as tool message
        error_message = ToolMessage(
            content=f"Tool execution failed: {str(e)}",
            name="error",
            tool_call_id=f"tool_{thread_id}_error",
            metadata={"source": "tool_executor", "error": True}
        )
        return {"messages": messages + [error_message]}