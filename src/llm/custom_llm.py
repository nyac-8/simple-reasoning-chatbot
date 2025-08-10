"""
Custom LLM wrapper using langchain-google-genai.
Provides a lean interface for text generation, structured output, and tool calling.
"""

import os
import json
from typing import Any, Dict, List, Optional
from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class CustomLLM:
    """
    Lean LLM wrapper using langchain-google-genai.
    
    Core methods:
    1. generate_content: Text generation (str -> str)
    2. get_structured_output: JSON output with schema (str, dict -> str)
    3. tool_calls: Function calling with tools (str, tools -> str)
    """
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        **kwargs
    ):
        """Initialize with model configuration."""
        self.model_name = model
        self.temperature = temperature
        self.model = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            **kwargs
        )
        logger.info(f"CustomLLM initialized: model={model}, temp={temperature}")
    
    def generate_content(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt string
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text string
        """
        try:
            messages = [HumanMessage(content=prompt)]
            response = self.model.invoke(messages, **kwargs)
            return response.content
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def generate_from_messages(self, messages: List, **kwargs) -> str:
        """
        Generate text from a list of messages.
        
        Args:
            messages: List of message objects (in temporal order)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text string
        """
        try:
            response = self.model.invoke(messages, **kwargs)
            return response.content
        except Exception as e:
            logger.error(f"Generation from messages failed: {e}")
            raise
    
    def get_structured_output(
        self,
        prompt: str,
        schema: Dict[str, Any],
        **kwargs
    ) -> str:
        """
        Generate structured JSON output.
        
        Args:
            prompt: Input prompt string
            schema: JSON schema dict (Vertex AI format)
            **kwargs: Additional parameters
        
        Returns:
            JSON string
        """
        try:
            # Configure model for JSON output
            model = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                response_mime_type="application/json",
                response_schema=schema
            )
            
            messages = [HumanMessage(content=prompt)]
            response = model.invoke(messages, **kwargs)
            
            # Response is already JSON formatted
            return response.content
        except Exception as e:
            logger.error(f"Structured generation failed: {e}")
            raise
    
    def get_structured_output_with_messages(
        self,
        messages: List,
        system_prompt: str,
        schema: Dict[str, Any],
        **kwargs
    ) -> str:
        """
        Generate structured JSON output with message history.
        
        Args:
            messages: List of messages (temporal order)
            system_prompt: System prompt to prepend
            schema: JSON schema dict (Vertex AI format)
            **kwargs: Additional parameters
        
        Returns:
            JSON string
        """
        try:
            # Configure model for JSON output
            model = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                response_mime_type="application/json",
                response_schema=schema
            )
            
            # Build full message list with system prompt
            from langchain_core.messages import SystemMessage
            full_messages = [SystemMessage(content=system_prompt)] + messages
            
            response = model.invoke(full_messages, **kwargs)
            
            # Response is already JSON formatted
            return response.content
        except Exception as e:
            logger.error(f"Structured generation with messages failed: {e}")
            raise
    
    def tool_calls(
        self,
        prompt: str,
        tools: List[BaseTool],
        **kwargs
    ) -> str:
        """
        Generate with tool calling.
        
        Args:
            prompt: Input prompt string
            tools: List of LangChain BaseTool objects
            **kwargs: Additional parameters
        
        Returns:
            JSON string with response and tool calls
        """
        try:
            # Bind tools to model
            model_with_tools = self.model.bind_tools(tools)
            
            messages = [HumanMessage(content=prompt)]
            response = model_with_tools.invoke(messages, **kwargs)
            
            result = {
                "text": response.content,
                "tool_calls": []
            }
            
            # Extract tool calls if present
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    result["tool_calls"].append({
                        "name": tool_call["name"],
                        "args": tool_call["args"]
                    })
            
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Tool calling failed: {e}")
            raise


"""
================================================================================
USAGE GUIDE & EXAMPLES FOR CUSTOMLLM
================================================================================

This guide provides comprehensive examples and findings from testing Gemini's
capabilities through the CustomLLM wrapper.

--------------------------------------------------------------------------------
1. BASIC TEXT GENERATION
--------------------------------------------------------------------------------

Simple text generation with various temperature settings:

```python
from src.llm.custom_llm import CustomLLM

# Initialize with low temperature for deterministic outputs
llm = CustomLLM(model="gemini-2.0-flash", temperature=0.0)

# Basic question answering
response = llm.generate_content("What is 2 + 2?")
# Returns: "2 + 2 = 4"

# Creative generation with higher temperature
creative_llm = CustomLLM(model="gemini-2.0-flash", temperature=0.7)
poem = creative_llm.generate_content("Write a haiku about programming")
# Returns: "Code lines take their form,\nLogic blooms in silent grace,\nBugs hide, then appear."
```

--------------------------------------------------------------------------------
2. STRUCTURED JSON OUTPUT
--------------------------------------------------------------------------------

Generate validated JSON matching a schema:

```python
# Decision-making schema
decision_schema = {
    "type": "object",
    "properties": {
        "thinking": {"type": "string"},
        "action": {"type": "string", "enum": ["think", "answer"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "ready_to_answer": {"type": "boolean"}
    },
    "required": ["thinking", "action", "confidence", "ready_to_answer"]
}

prompt = "Analyze the question 'What is the capital of France?' and decide if you need more thinking"
json_response = llm.get_structured_output(prompt, decision_schema)
# Returns JSON string: 
# {
#   "thinking": "The question is straightforward and requires no further analysis.",
#   "action": "answer",
#   "confidence": 100,
#   "ready_to_answer": true
# }

# Math problem solving schema
math_schema = {
    "type": "object",
    "properties": {
        "equation": {"type": "string"},
        "steps": {"type": "array", "items": {"type": "string"}},
        "solution": {"type": "number"},
        "verification": {"type": "string"}
    },
    "required": ["equation", "steps", "solution", "verification"]
}

math_response = llm.get_structured_output(
    "Solve the equation: 2x + 5 = 13",
    math_schema
)
# Returns properly structured solution with steps
```

--------------------------------------------------------------------------------
3. TOOL CALLING
--------------------------------------------------------------------------------

### Basic Tool Definition and Usage:

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")

@tool
def calculator(expression: str) -> str:
    "Perform mathematical calculations"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

# Single tool call
tools = [calculator]
response = llm.tool_calls(
    "What is 15 * 23 + 47?",
    tools
)
# Returns: {"text": "", "tool_calls": [{"name": "calculator", "args": {"expression": "15 * 23 + 47"}}]}
```

### Parallel Tool Calling:

```python
@tool
def get_weather(city: str) -> str:
    "Get weather for a city"
    weather_data = {
        "Paris": "22°C, Sunny",
        "London": "18°C, Cloudy"
    }
    return weather_data.get(city, "Unknown city")

@tool
def send_email(to: str, subject: str, body: str) -> str:
    "Send an email"
    return f"Email sent to {to}"

# Multiple tools called in parallel
tools = [calculator, get_weather, send_email]
response = llm.tool_calls(
    "Calculate 15*20, get weather for Paris and London, and email the results to test@example.com",
    tools
)
# Returns multiple tool calls executed in parallel:
# {
#   "text": "",
#   "tool_calls": [
#     {"name": "calculator", "args": {"expression": "15*20"}},
#     {"name": "get_weather", "args": {"city": "Paris"}},
#     {"name": "get_weather", "args": {"city": "London"}},
#     {"name": "send_email", "args": {...}}
#   ]
# }
```

### Tool Choice Configurations:

```python
# AUTO mode (default) - Model decides when to use tools
model_with_tools = llm.model.bind_tools(tools)

# ANY mode - Force tool use even for simple greetings
model_with_tools = llm.model.bind_tools(tools, tool_choice="any")

# NONE mode - Disable tools temporarily
model_with_tools = llm.model.bind_tools(tools, tool_choice="none")
```

--------------------------------------------------------------------------------
4. COMPLEX NESTED ARGUMENTS
--------------------------------------------------------------------------------

Tools can handle complex nested data structures:

```python
from typing import Dict, List, Any, Optional

class ComplexQueryInput(BaseModel):
    filters: Dict[str, Any] = Field(description="Query filters")
    sorting: List[Dict[str, str]] = Field(description="Sort criteria")
    pagination: Dict[str, int] = Field(description="Pagination settings")
    options: Optional[Dict[str, bool]] = Field(default=None)

@tool
def complex_query(
    filters: Dict[str, Any],
    sorting: List[Dict[str, str]],
    pagination: Dict[str, int],
    options: Optional[Dict[str, bool]] = None
) -> str:
    "Execute complex query"
    return f"Query with {len(filters)} filters"

# Model correctly structures nested arguments
response = llm.tool_calls(
    "Query users where age > 25 and city = 'Paris', sort by name asc then date desc, page 2 with 20 items",
    [complex_query]
)
# Produces properly nested JSON arguments
```

--------------------------------------------------------------------------------
5. MULTI-TURN CONVERSATIONS WITH TOOLS
--------------------------------------------------------------------------------

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Build conversation with tool interactions
messages = []

# Turn 1: User asks for calculation
messages.append(HumanMessage(content="What's 150 * 23?"))
response = model_with_tools.invoke(messages)
messages.append(response)

# Execute tool and add result
if response.tool_calls:
    for tc in response.tool_calls:
        result = calculator.invoke(tc['args'])
        messages.append(ToolMessage(
            content=result,
            tool_call_id=tc.get('id', 'call_1')
        ))

# Model provides final answer incorporating tool result
final_response = model_with_tools.invoke(messages)
# "150 * 23 is 3450"

# Turn 2: Follow-up using previous context
messages.append(HumanMessage(content="Now add 500 to that"))
# Model understands context and calls calculator with "3450 + 500"
```

--------------------------------------------------------------------------------
6. KEY FINDINGS & LIMITATIONS
--------------------------------------------------------------------------------

### Tool Choice Formats:
- Supported: "auto", "any", "none"
- NOT supported: OpenAI-style specific tool selection
- Workaround: Use prompt engineering to guide tool selection

### Model Behavior:
- The model may refuse to call functions it predicts will fail
- Example: Won't divide by zero even when explicitly asked
- Solution: Frame requests positively or handle errors gracefully

### Parallel Execution:
- Model intelligently batches related tool calls
- Can execute multiple tools in single request
- Reduces latency for complex operations

### Optional Parameters:
- Model correctly identifies which parameters are optional
- Only includes necessary parameters in tool calls
- Handles default values appropriately

### Error Handling Pattern:
```python
try:
    response = model_with_tools.invoke(prompt)
    if response.tool_calls:
        for tc in response.tool_calls:
            try:
                result = tool.invoke(tc['args'])
                # Process successful result
            except Exception as tool_error:
                # Send error back to model for recovery
                error_msg = ToolMessage(
                    content=f"Error: {str(tool_error)}",
                    tool_call_id=tc.get('id', 'error_id')
                )
                recovery = model_with_tools.invoke([prompt, response, error_msg])
except Exception as e:
    logger.error(f"Model invocation failed: {e}")
```

### Best Practices:
1. Use temperature=0 for deterministic outputs
2. Validate JSON schemas before using structured output
3. Implement proper error handling for tool failures
4. Cache model instances to avoid re-initialization
5. Log all LLM interactions for debugging
6. Use descriptive tool names and clear descriptions
7. Keep tool signatures simple when possible

### Common Pitfalls:
1. Don't assume specific tool choice format compatibility
2. Model may be overly cautious with risky operations
3. Complex nested arguments may be stringified
4. Rate limits can interrupt batch processing
5. Tool descriptions affect selection accuracy

================================================================================
"""