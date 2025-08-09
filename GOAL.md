# Simple Reasoning Chatbot - Goals

## Project Overview
Build a LangGraph-based orchestration layer that emulates the "thinking → answering" pattern seen in models like o1/Claude, using Gemini 2.0 Flash exclusively.

## Version 1 Objectives (COMPLETED)

### Core Functionality
- **Agentic Q&A System**: General-purpose question answering with visible reasoning steps
- **Dynamic Reasoning**: Flexible number of reasoning steps based on question complexity
- **Self-Terminating**: System decides when reasoning is sufficient to answer
- **Session Management**: Support multi-turn conversations with proper history tracking

### Architecture

#### Graph Structure
- **Two Main Nodes/Agents**:
  1. **Orchestrator Agent**: Handles thinking/reasoning loops, decides when ready to answer
  2. **Writer Agent**: Formats final response from reasoning context

#### State Design
```python
{
    "session_id": str,              # Conversation identifier
    "thread_id": str,               # Single Q&A interaction identifier
    "messages": List[BaseMessage],  # Current thread messages with history
    "history": List[Tuple[str, str]], # [(question, answer)...] for context
    "reasoning_steps": List[AIMessage], # Internal reasoning (type="reasoning")
    "context": Dict[str, Any],      # Empty in v1, for tools in v2 (doc_ids, etc.)
    "tools": List[Tool],            # None/[] in v1, populated in v2
    "ready_to_answer": bool,        # Reasoner's decision flag
    "final_answer": Optional[str]   # Writer's output
}
```

### Message Types
- `HumanMessage`: User questions
- `AIMessage(type="reasoning")`: Thinking steps (internal)
- `AIMessage(type="final_answer")`: Clean response
- `SystemMessage`: Prompts/instructions

### Constraints
- **Single LLM**: Gemini-2.0-flash-text only
- **Sync Only**: Use `.invoke()` only (no async/streaming)
- **Notebook First**: Jupyter notebook environment
- **Simple**: No subgraphs, no parallelization

### Version 1 Deliverables
1. Working reasoning loop with emergent steps
2. Clean separation between reasoning and writing
3. Proper conversation history management
4. State architecture ready for v2 tools extension
5. Clear abstractions (node = agent)

### Success Criteria
- Can answer simple and complex questions with visible reasoning
- Reasoning steps adapt to question complexity
- Clean, formatted final answers
- Maintains conversation context across threads
- Architecture easily extendable for v2 tools

## Version 2 Objectives (CURRENT)

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
- **REMOVE WRITER_TEMPERATURE**: No hardcoded temperature constants

#### 2. Simplify State Management
- **REMOVE reasoning_steps field**: No separate reasoning_steps list
- **REMOVE reasoning_count field**: Count dynamically from messages
- **Messages field becomes single source of truth**: Contains ALL messages for current thread

#### 3. Message Type Constraints
- **Messages MUST contain ONLY**:
  - `HumanMessage`: User inputs
  - `AIMessage`: Both reasoning and final answers
- **Use metadata/kwargs for differentiation**:
  - `type`: "reasoning" | "final_answer"
  - `source`: "orchestrator" | "writer"
  - Other metadata as needed
- **NO SystemMessage in state.messages**: System prompts/history injected at invoke time only

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

#### 6. Clean Invocation Pattern
```python
# Build prompt outside state
full_messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    *format_history(state["history"]),
    *state["messages"]  # Clean, unpolluted
]
# Invoke with composed messages
response = model.invoke(full_messages)
# Return only the new message
return {"messages": state["messages"] + [response]}
```

### Core Enhancement: Tool-Augmented Reasoning
Transform the pure reasoning system into a tool-augmented intelligence that can execute code, perform operations, and search for real-time information.

### New Tools Integration

#### Tool Architecture
- **LangChain Tool Format**: All tools implemented as LangChain `BaseTool` with proper schemas
- **Tool Messages**: Tools return `ToolMessage` objects following LangChain conventions
- **Hardcoded Integration**: Tools are directly imported and bound to orchestrator (no dynamic loading)
- **Directory Structure**: New `src/tools/` directory containing all tool implementations

#### Three Core Tools

1. **REPL Tool** (`src/tools/repl.py`)
   - Execute Python code in isolated environment
   - Run calculations, test algorithms, data analysis
   - Returns code output, stdout, and any errors
   - Stateful execution context within a thread

2. **Operator Tool** (`src/tools/operator.py`)
   - Structured operations for math, logic, data transforms
   - Handles symbolic math, unit conversions, logical operations
   - More constrained than REPL, but safer and faster
   - Returns structured operation results

3. **Tavily Search Tool** (`src/tools/tavily_search.py`)
   - Web search for current information and fact-checking
   - Uses Tavily API (requires TAVILY_API_KEY in .env)
   - Returns search results with snippets and sources
   - Configurable search depth and result count

### Enhanced Orchestrator v2

#### Decision Schema
```json
{
  "thinking": "Current reasoning thought process",
  "action": "think|repl|operator|search|answer",
  "tool_name": "repl|operator|tavily_search",
  "tool_input": {
    "code": "...",  // for REPL
    "operation": "...",  // for Operator
    "query": "..."  // for Tavily
  },
  "confidence": 85,
  "missing_info": ["what still needs investigation"]
}
```

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
    "session_id": str,
    "thread_id": str,
    "messages": List[BaseMessage],
    "history": List[Tuple[str, str]],
    "reasoning_steps": List[BaseMessage],
    "tool_history": List[ToolMessage],  # NEW: Track tool usage
    "context": Dict[str, Any],          # NOW USED: Store tool outputs
    "tools": List[BaseTool],            # NOW POPULATED: Available tools
    "ready_to_answer": bool,
    "final_answer": Optional[str],
    "current_question": str,
    "reasoning_count": int,
    "tool_count": int                   # NEW: Track tool invocations
}
```

### Graph Flow v2
```
User Question 
    ↓
Orchestrator (assess)
    ↓
[Decision Point]
    ├─→ Think More (loop back)
    ├─→ Use Tool → Tool Execution → Orchestrator (re-assess)
    └─→ Ready → Writer → Final Answer
```

### Tool Integration Implementation

#### Directory Structure
```
src/
├── tools/
│   ├── __init__.py
│   ├── repl.py         # Python REPL tool
│   ├── operator.py     # Structured operations tool
│   └── tavily_search.py # Web search tool
├── agents/
│   ├── orchestrator.py  # Updated with tool binding
│   └── writer.py        # Now references tool outputs
```

#### Tool Binding in Orchestrator
```python
# Hardcoded tool initialization
from ..tools import REPLTool, OperatorTool, TavilySearchTool

tools = [
    REPLTool(),
    OperatorTool(), 
    TavilySearchTool()
]
# Tools are bound to model via .bind_tools()
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

### Next Steps When Resuming
1. Create `src/tools/` directory structure
2. Implement the three tools as LangChain BaseTool:
   - Start with TavilySearchTool (simplest, API-based)
   - Then OperatorTool (structured operations)
   - Finally REPLTool (most complex, needs sandbox)
3. Update State TypedDict with new fields (tool_history, tool_count)
4. Modify orchestrator.py:
   - Import and initialize tools
   - Update decision logic for tool selection
   - Add tool execution flow
5. Update prompts.py with tool-aware orchestrator prompt
6. Enhance writer.py to reference tool outputs
7. Create demo notebook for v2 testing

### Key Design Decisions Made
- Tools are hardcoded imports (not dynamic)
- All tools follow LangChain conventions
- Orchestrator uses JSON schema for decisions
- Tool results stored in state for writer access
- MAX_REASONING_STEPS increased to 30-40 for v2

## Version 3+ Future Roadmap
- Document processing tools (RAG, QA)
- Multi-agent systems with specialized agents
- Parallel tool execution
- Custom tool creation interface
- Long-term memory and knowledge base