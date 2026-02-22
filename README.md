# Multi-Agent Collaboration Execution Engine

A LangGraph-based multi-agent collaboration engine with Planner/Executor/Critic architecture, MCP tool registry, state persistence, vector memory, and FastAPI interface.

## Features

- **Planner/Executor/Critic Architecture**: Hierarchical agent design for complex task decomposition and execution
- **MCP Tool Registry**: Model Context Protocol-based tool management with dynamic registration
- **State Persistence**: PostgreSQL-based task state and checkpoint storage for interrupt/resume
- **Vector Memory**: Long-term memory using pgvector for context-aware execution
- **LLM Provider**: Minimax LLM integration
- **Token Budget Control**: Execution limits for tokens, steps, and tool calls
- **FastAPI Interface**: REST API and WebSocket support for external integration

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (for PostgreSQL with pgvector)
- Poetry (for dependency management)

## Quick Start

1. **Clone and setup environment**
   ```bash
   cd agent_demo
   cp .env.example .env
   # Edit .env with your Minimax API key and Group ID
   ```

2. **Start PostgreSQL with pgvector**
   ```bash
   docker-compose up -d
   ```

3. **Install dependencies**
   ```bash
   poetry install
   ```

4. **Run database migrations**
   ```bash
   poetry run alembic upgrade head
   ```

5. **Start the server**
   ```bash
   poetry run agent-engine
   ```

   Or with uvicorn directly:
   ```bash
   poetry run uvicorn agent_engine.main:app --reload
   ```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tasks` | Create a new task |
| GET | `/tasks/{id}` | Get task status and details |
| POST | `/tasks/{id}/resume` | Resume an interrupted task |
| DELETE | `/tasks/{id}` | Cancel a running task |
| WS | `/tasks/{id}/stream` | Real-time execution stream |
| GET | `/tools` | List available tools |
| POST | `/tools` | Register a new tool |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Layer                           │
│                  (REST + WebSocket)                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  LangGraph Orchestrator                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Planner  │───▶│ Executor │───▶│  Critic  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────┬───────────┴───────────┬───────────────────────┐
│ MCP Tools   │  Memory System        │  State Persistence    │
│ Registry    │  (pgvector)           │  (PostgreSQL)         │
└─────────────┴───────────────────────┴───────────────────────┘
```

## Configuration

All configuration is done via environment variables. See `.env.example` for available options.

### Execution Limits

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_TOKENS` | 100,000 | Maximum tokens per task |
| `MAX_STEPS` | 50 | Maximum execution steps |
| `MAX_TOOL_CALLS` | 100 | Maximum tool invocations |
| `EXECUTION_TIMEOUT` | 600 | Timeout in seconds |

## Development

```bash
# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=agent_engine

# Lint code
poetry run ruff check src/
```

## License

MIT
