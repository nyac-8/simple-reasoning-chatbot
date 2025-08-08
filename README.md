# Simple Reasoning Chatbot

A LangGraph-based orchestration system that emulates the "thinking → answering" pattern seen in advanced AI models, using Gemini 2.0 Flash.

## Features

- **Dynamic Reasoning**: Flexible number of reasoning steps based on question complexity
- **Self-Terminating**: System decides when reasoning is sufficient
- **Session Management**: Multi-turn conversations with proper history tracking
- **Real-time Display**: Watch reasoning unfold as it happens
- **Colored Output**: Visual distinction between reasoning (yellow) and answers (green)

## Architecture

```
User Question → Orchestrator (reasoning loop) → Writer (final answer) → Response
```

- **Orchestrator Agent**: Handles thinking/reasoning loops, decides when ready to answer
- **Writer Agent**: Formats final response from reasoning context

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
# or with uv:
uv sync
```

2. Set up your Gemini API key in `.env`:
```
GEMINI_API_KEY=your-api-key-here
GOOGLE_API_KEY=your-api-key-here
```

3. Run the demo notebook:
```bash
jupyter notebook notebooks/demo.ipynb
```

## Project Structure

```
src/
├── state.py           # State schema definition
├── graph.py           # LangGraph workflow
├── prompts.py         # System prompts
├── utils.py           # Logging utilities  
└── agents/
    ├── orchestrator.py  # Reasoning agent
    └── writer.py        # Response formatting agent
```

## Version 1.0 Features

- JSON-based reasoning decisions
- Self-reflective reasoning style
- Guardrails for safe responses
- Advanced prompting techniques (CoT, ToT concepts)
- Thread lifecycle visibility
- No external tools (pure reasoning)

## Future (Version 2+)

- Tools integration (web search, document QA, RAG)
- Multi-agent systems
- Parallel execution
- Enhanced context management

## License

MIT
