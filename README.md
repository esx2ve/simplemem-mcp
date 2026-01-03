# simplemem-mcp

Lightweight MCP client for SimpleMem - long-term memory for Claude Code.

## Installation

```bash
# Add to Claude Code
claude mcp add simplemem --command "uvx simplemem-mcp serve"

# Or with API key
claude mcp add simplemem --command "uvx simplemem-mcp serve" \
  --env SIMPLEMEM_API_KEY=<your-key>
```

## Commands

```bash
simplemem-mcp serve    # Run MCP server (for Claude Code)
simplemem-mcp signup   # Get an API key
simplemem-mcp config   # Show configuration
simplemem-mcp version  # Show version
```

## Features

- **Hybrid Search**: Vector similarity + graph traversal
- **Session Traces**: Auto-index Claude Code sessions
- **Code Index**: Semantic code search with file watching
- **Cross-Session Learning**: Patterns discovered across sessions

## Architecture

```
simplemem-mcp (this package)     simplemem-lite (cloud backend)
┌─────────────────────┐          ┌─────────────────────────┐
│  Thin MCP Client    │  HTTPS   │  Fly.io                 │
│  - stdio protocol   │ ───────► │  - LanceDB (vectors)    │
│  - local file read  │          │  - KuzuDB (graph)       │
│  - gzip compression │          │  - LiteLLM (embeddings) │
└─────────────────────┘          └─────────────────────────┘
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMPLEMEM_BACKEND_URL` | `https://simplemem-lite.fly.dev` | Backend API URL |
| `SIMPLEMEM_API_KEY` | - | API key for authentication |

## License

MIT
