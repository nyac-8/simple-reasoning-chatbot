# Simple Reasoning Chatbot - Goals

## Project Overview
Build a LangGraph-based orchestration layer that emulates the "thinking → answering" pattern seen in models like o1/Claude, using Gemini 2.0 Flash exclusively.


## Version 2 Objectives (CURRENT)

### Graph Architecture Proposal

#### Core Flow Pattern
```
[START] → Orchestrator → [Think Loop OR Tool Call OR Write]
                ↑                    ↓
                └──────────────── ToolNode
                                     
Orchestrator → Writer → [END]
```

#### Node Definitions
1. **Orchestrator Node** (ReAct-inspired)
   - Always performs at least ONE think step (1+ pattern)
   - Decides next action: think_more | use_tool | write_answer
   - Returns: `{"messages": [...], "ready_to_answer": bool}`
   - Tool calls return here for post-tool reasoning

2. **ToolNode** (LangGraph's built-in)
   - Executes tool calls from orchestrator
   - Handles REPL and Tavily Search tools
   - Returns ToolMessage to messages
   - Always routes back to orchestrator for thinking

3. **Writer Node**
   - Formats final response from reasoning + tool context
   - Returns: `{"messages": [final_answer]}`

#### Valid Execution Paths
- **Minimal**: START → Orchestrator(think) → Writer → END
- **Multi-think**: START → Orchestrator(think) → Orchestrator(think) → ... → Writer → END  
- **Tool-augmented**: START → Orchestrator(think) → ToolNode → Orchestrator(think) → Writer → END
- **Complex**: START → Orchestrator(think) → ToolNode → Orchestrator(think) → ToolNode → Orchestrator(think) → Writer → END

#### Key Constraints
- **Minimum one think**: Every response requires ≥1 reasoning step
- **Post-tool think**: After any tool execution, must return to orchestrator for reasoning
- **No forced patterns**: Orchestrator decides organically when tools are needed
- **Self-terminating**: Orchestrator decides when reasoning is sufficient

### CRITICAL: Functional Programming & LangGraph Best Practices

#### Core Principles
- **Immutable State**: Never mutate state directly. Nodes return new values, not modified objects
- **Pure Functions**: Nodes are pure functions: (state, config) → partial_state
- **Single Responsibility**: Each node returns ONLY the fields it's responsible for
- **Framework Handles Merging**: LangGraph's reducers handle state merging automatically

#### Node Design Pattern
```python
def node(state: State, config: Config) -> PartialState:
    # 1. Read from state (immutable)
    current_messages = state["messages"]
    
    # 2. Perform computation (pure)
    new_message = process_logic(current_messages)
    
    # 3. Return ONLY changed fields
    return {"messages": current_messages + [new_message]}
    # NOT: return full state or mutate state
```

#### Key Rules
1. **Once Modified, Don't Touch**: If a node modifies a field, it shouldn't build further on that field within the same execution
2. **Return Partial State**: Return only the fields you're updating, let reducers handle merging
3. **No Side Effects in State Logic**: Side effects (API calls, file I/O) are OK, but state transformation must be pure
4. **Checkpointing Friendly**: Immutable patterns enable time-travel debugging and state replay

### Pre-Implementation Architecture Changes
**MUST BE COMPLETED BEFORE ADDING TOOLS:**

#### 1. Remove Static Constraints
- **REMOVE MIN_REASONING_LENGTH**: No minimum reasoning steps requirement
- **Temperature at model init**: Set temperature=0 at model initialization (no constants)

#### 2. Simplify State Management
- **REMOVE reasoning_steps field**: No separate reasoning_steps list
- **REMOVE reasoning_count field**: Count dynamically from messages
- **REMOVE current_question field**: Extract from first HumanMessage in thread
- **Messages field becomes single source of truth**: Contains ALL messages for current THREAD
  - Thread = one complete Q&A interaction with reasoning
  - Session history managed externally (first human + final answer pairs)

#### 3. Message Type Constraints
- **Messages MUST contain ONLY**:
  - `HumanMessage`: User inputs
  - `AIMessage`: Both reasoning and final answers
  - `ToolMessage`: Tool execution results (v2)
- **Use metadata for differentiation**:
  - `type`: "reasoning" | "final_answer" | "tool_result"
  - `source`: "orchestrator" | "writer" | "tool"
  - Other metadata as needed
- **NO SystemMessage in state.messages**: System prompts/history injected at invoke time only
- **NO thread_id/session_id in messages**: These are implied from context

#### 4. Node Responsibilities (Strict Functional Pattern)
- **Orchestrator Node**:
  - READS: messages, context, tools
  - RETURNS: `{"messages": [...new_reasoning], "ready_to_answer": bool}`
  - NEVER: Modifies then re-reads messages in same execution
- **Writer Node**:
  - READS: messages (with all reasoning)
  - RETURNS: `{"messages": [...final_answer]}`
  - NEVER: Touches ready_to_answer or any other field

#### 5. Dynamic Reasoning Counting
```python
# Count dynamically when needed
reasoning_count = len([m for m in state["messages"] 
                       if isinstance(m, AIMessage) 
                       and m.metadata.get('type') == 'reasoning'])
```

#### 6. Prompt Composition Strategy (Future-Proof)
**Critical**: Prepare for custom LLMs that only accept string prompts

```python
# Current: LangChain ChatModel interface
model.invoke(messages)  # List[BaseMessage]

# Future: Custom LLM interface
def compose_prompt(messages: List[BaseMessage]) -> str:
    """Convert messages to single prompt string for custom LLMs"""
    prompt_parts = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            prompt_parts.append(f"System: {msg.content}")
        elif isinstance(msg, HumanMessage):
            prompt_parts.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            prompt_parts.append(f"Assistant: {msg.content}")
        elif isinstance(msg, ToolMessage):
            prompt_parts.append(f"Tool: {msg.content}")
    return "\n\n".join(prompt_parts)

# In nodes:
full_messages = [SystemMessage(PROMPT), *state["messages"]]
if USE_CUSTOM_LLM:
    prompt_str = compose_prompt(full_messages)
    response = custom_llm.generate(prompt_str)
else:
    response = langchain_model.invoke(full_messages)
```

#### 7. Clean Invocation Pattern
```python
# Build prompt outside state
full_messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    *format_history(state["history"]),  # Only if needed
    *state["messages"]  # Clean, unpolluted thread messages
]
# Invoke with composed messages
response = model.invoke(full_messages)
# Return only the new message with metadata
new_message = AIMessage(
    content=response.content,
    metadata={"type": "reasoning", "source": "orchestrator"}
)
return {"messages": state["messages"] + [new_message]}
```

### Core Enhancement: Tool-Augmented Reasoning
Transform the pure reasoning system into a tool-augmented intelligence that can execute code, perform operations, and search for real-time information.

### New Tools Integration

#### Tool Architecture
- **LangChain Tool Format**: All tools implemented as LangChain `BaseTool` with proper schemas
- **Tool Messages**: Tools return `ToolMessage` objects following LangChain conventions
- **Hardcoded Integration**: Tools are directly imported and bound to orchestrator (no dynamic loading)
- **Directory Structure**: New `src/tools/` directory containing all tool implementations

#### Two Core Tools (Simple Implementation)

1. **REPL Tool** (`src/tools/repl.py`)
   - Uses existing `PythonREPL` from `langchain_experimental.utilities`
   - Simple wrapper as LangChain `BaseTool`
   - Execute Python code for calculations and data analysis
   - Returns code output or errors
   - Minimal custom code, leverage community tools

2. **Tavily Search Tool** (`src/tools/tavily_search.py`)
   - Uses existing `TavilySearchResults` from `langchain_community.tools`
   - Simple configuration wrapper
   - Web search for current information
   - Requires TAVILY_API_KEY in .env
   - Returns search results with snippets

### Enhanced Orchestrator v2 (ReAct Pattern)

#### Decision Flow
```python
# Orchestrator internally decides:
if needs_more_thinking:
    return {"messages": [reasoning_msg], "ready_to_answer": False}
elif needs_tool:
    # Generate tool call message
    return {"messages": [tool_call_msg], "ready_to_answer": False}
else:  # ready to answer
    return {"messages": messages, "ready_to_answer": True}
```

#### Tool Calling Pattern
- Orchestrator generates AIMessage with tool_calls
- ToolNode executes and adds ToolMessage
- Control returns to orchestrator for reasoning about results
- Natural ReAct flow without rigid structure

#### Tool Selection Logic
- Pattern matching on user query keywords
- Confidence-based tool selection
- Tool chaining capability (use multiple tools in sequence)
- Fallback to pure reasoning if tools fail

#### Increased Complexity Handling
- MAX_REASONING_STEPS increased to 30-40
- Dynamic step allocation based on tool usage
- Tool results integrated into reasoning context

### State Schema v2 Updates
```python
{
    "session_id": str,                  # Session identifier (managed externally)
    "thread_id": str,                   # Thread identifier (one Q&A cycle)
    "messages": List[BaseMessage],      # ALL messages in thread (Human, AI, Tool)
    "history": List[Tuple[str, str]],   # Past threads (first human + final answer)
    "context": Dict[str, Any],          # Tool outputs, search results, etc.
    "tools": List[BaseTool],            # Available tools for v2
    "ready_to_answer": bool,            # Orchestrator's decision flag
    "final_answer": Optional[str]       # Cached final answer for convenience
}
```

**Key Changes from v1**:
- REMOVED: `reasoning_steps` (now in messages with metadata)
- REMOVED: `reasoning_count` (count dynamically from messages)
- REMOVED: `current_question` (extract from first HumanMessage)
- REMOVED: `tool_history` (tools write to messages directly)
- SIMPLIFIED: Everything flows through messages with clear metadata

### Graph Implementation v2

#### LangGraph Setup
```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

graph = StateGraph(State)

# Add nodes
graph.add_node("orchestrator", orchestrator_node)
graph.add_node("tools", ToolNode(tools=[repl_tool, tavily_tool]))
graph.add_node("writer", writer_node)

# Add edges
graph.add_edge(START, "orchestrator")
graph.add_conditional_edges(
    "orchestrator",
    should_continue,  # Checks ready_to_answer and tool_calls
    {
        "continue": "orchestrator",  # Think more
        "tools": "tools",            # Use tool
        "writer": "writer"           # Ready to answer
    }
)
graph.add_edge("tools", "orchestrator")  # Always back to think
graph.add_edge("writer", END)
```

#### Routing Logic
```python
def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    
    # Check if ready to answer
    if state.get("ready_to_answer"):
        return "writer"
    
    # Check if tool call needed
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Continue thinking
    return "continue"
```

### Tool Integration Implementation

#### Directory Structure
```
src/
├── tools/
│   ├── __init__.py
│   ├── repl.py         # Python REPL tool wrapper
│   └── tavily_search.py # Tavily search wrapper
├── agents/
│   ├── orchestrator.py  # Updated with tool binding
│   └── writer.py        # References tool outputs
```

#### Tool Setup (Minimal Code)
```python
# src/tools/repl.py
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import BaseTool

class REPLTool(BaseTool):
    name = "python_repl"
    description = "Execute Python code"
    
    def _run(self, code: str) -> str:
        repl = PythonREPL()
        return repl.run(code)

# src/tools/tavily_search.py  
from langchain_community.tools import TavilySearchResults

def get_tavily_tool():
    return TavilySearchResults(max_results=3)

# In orchestrator
tools = [REPLTool(), get_tavily_tool()]
model_with_tools = model.bind_tools(tools)
```

### Success Criteria v2
- Orchestrator correctly selects tools based on query type
- Tools execute and return proper ToolMessage responses
- Tool results enhance reasoning quality
- System handles tool failures gracefully
- Clear distinction between tool-augmented and pure reasoning paths
- Maintains all v1 capabilities while adding tool power

### Constraints v2
- Still using Gemini-2.0-flash exclusively
- Still using .invoke() only (no async/streaming)
- Tools are synchronous LangChain tools
- No dynamic tool loading (hardcoded set)
- Tool timeout limits to prevent hanging

## Implementation Notes for Next Session

### Where We Left Off
- Version 1 is complete and merged to master
- Version 2 branch created and GOAL.md updated with tool specifications
- Environment has TAVILY_API_KEY ready

### Implementation Guidelines for v2

#### Critical Rules
1. **NO v2 scripts**: Update existing files only (no `orchestrator_v2.py`, etc.)
   - Modify existing `src/` files in-place
   - Git commits will memorialize as v2
   
2. **Test with temp scripts**: 
   - Create temporary `test_*.py` scripts for testing
   - Delete temp scripts after validation
   - Don't commit test scripts
   
3. **Update demo notebook**:
   - Update `notebooks/demo.ipynb` with v2 capabilities
   - Show tool usage examples
   - Demonstrate all execution paths

### Next Steps When Resuming
1. Implement minimal tool wrappers:
   - `src/tools/repl.py`: Wrap PythonREPL as BaseTool
   - `src/tools/tavily_search.py`: Configure TavilySearchResults
2. Update graph.py:
   - Import ToolNode from langgraph.prebuilt
   - Add tools node to graph
   - Implement should_continue routing logic
3. Modify orchestrator.py:
   - Bind tools to model
   - Natural reasoning that can generate tool calls
   - Ensure 1+ think pattern
4. Update prompts.py with tool-aware orchestrator prompt
5. Test execution paths:
   - Human → Think → Writer (minimal)
   - Human → Think → Tool → Think → Writer (tool-augmented)
6. Create demo notebook for v2 testing

### Key Design Decisions Made
- Tools use existing community implementations (minimal custom code)
- LangGraph's ToolNode handles tool execution
- ReAct-inspired pattern without forcing structure
- Minimum 1 think step before any answer (1+ pattern)
- Tool calls always followed by orchestrator reasoning
- Graph has clear conditional routing based on state

## Version 3+ Future Roadmap
- Document processing tools (RAG, QA)
- Multi-agent systems with specialized agents
- Parallel tool execution
- Custom tool creation interface
- Long-term memory and knowledge base