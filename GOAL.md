# Simple Reasoning Chatbot - Version 1 Goals

## Project Overview
Build a LangGraph-based orchestration layer that emulates the "thinking → answering" pattern seen in models like o1/Claude, using Gemini 2.0 Flash exclusively.

## Version 1 Objectives

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

## Future Ready (Not in v1 scope)
- **Version 2**: Add tools (qa_with_full_document, rag_on_documents, web_search)
- **Context Usage**: `context` dict will store doc_ids, search params, etc.
- **Tool Integration**: Tools consume context to enrich thread
- **Version 3+**: Multi-agent systems, parallel execution