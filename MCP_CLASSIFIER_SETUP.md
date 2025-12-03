# MCP Classifier Setup Guide

## Overview

The question classifier can be run as a separate MCP (Model Context Protocol) server, allowing for better separation of concerns and the ability to scale the classification service independently.

## Installation

### 1. Install MCP SDK

```bash
pip install "mcp[cli]"
```

Or if using requirements.txt:

```bash
pip install -r requirements.txt
```

## Running the MCP Server

### Option 1: Run as Separate Process (Recommended)

Start the MCP classifier server:

```bash
python mcp_classifier_server.py
```

The server will start on `http://localhost:8001/mcp` by default.

### Option 2: Run with Custom Port

Set the port via environment variable:

```bash
MCP_CLASSIFIER_PORT=9000 python mcp_classifier_server.py
```

Or update `.env`:

```bash
MCP_CLASSIFIER_PORT=9000
```

## Configuration

### Enable MCP Classifier in Graph RAG

Update your `.env` file:

```bash
# Enable MCP classifier
USE_MCP_CLASSIFIER=true
MCP_CLASSIFIER_URL=http://localhost:8001/mcp
```

### Configuration Options

```bash
# Enable/disable MCP classifier
USE_MCP_CLASSIFIER=false              # Set to true to use MCP server

# MCP Server Configuration
MCP_CLASSIFIER_URL=http://localhost:8001/mcp
MCP_CLASSIFIER_PORT=8001
MCP_CLASSIFIER_TIMEOUT=30

# Classifier Behavior (used by MCP server)
CLASSIFIER_USE_HEURISTICS=true        # Fast heuristic classification
CLASSIFIER_USE_LLM=true               # LLM-based classification
```

## Testing

### Test MCP Server Connection

```bash
python mcp_classifier_client.py
```

This will:
1. Test connection to MCP server
2. List available tools
3. Test classification on sample questions

### Test from Graph RAG

When `USE_MCP_CLASSIFIER=true`, the Graph RAG system will automatically use the MCP server for classification. The system will fall back to direct classification if the MCP server is unavailable.

## Architecture

```
┌─────────────────────┐
│  Graph RAG App      │
│  (FastAPI)          │
└──────────┬──────────┘
           │
           │ HTTP/MCP Protocol
           │
┌──────────▼──────────┐
│  MCP Classifier     │
│  Server             │
│  (mcp_classifier_   │
│   server.py)        │
└──────────┬──────────┘
           │
           │ Uses LLM Client
           │
┌──────────▼──────────┐
│  LLM Service        │
│  (Ollama/OpenAI)    │
└─────────────────────┘
```

## Benefits of MCP Approach

1. **Separation of Concerns**: Classification logic is isolated
2. **Scalability**: Can scale classifier server independently
3. **Tool Calling**: Uses standard MCP protocol for tool calling
4. **Flexibility**: Can be replaced or upgraded without changing main app
5. **Fallback**: Main app falls back to direct classification if MCP unavailable

## Fallback Behavior

If the MCP classifier is enabled but unavailable:

1. The system will log a warning
2. Fall back to direct classification (heuristics + LLM)
3. Continue functioning normally

This ensures the system remains resilient.

## Production Deployment

### Running as Service

You can run the MCP server as a systemd service or using process managers like:

- **systemd** (Linux)
- **supervisord**
- **PM2** (Node.js process manager)

Example systemd service:

```ini
[Unit]
Description=MCP Question Classifier Server
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/graph-rag
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python mcp_classifier_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### Docker Deployment

You can also containerize the MCP server separately:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY mcp_classifier_server.py .
COPY utils.py .
COPY .env .

EXPOSE 8001

CMD ["python", "mcp_classifier_server.py"]
```

## Troubleshooting

### Server Won't Start

- Check if port 8001 is already in use
- Verify MCP SDK is installed: `pip list | grep mcp`
- Check logs for errors

### Client Can't Connect

- Verify server is running: `curl http://localhost:8001/mcp`
- Check `MCP_CLASSIFIER_URL` in `.env` matches server URL
- Check firewall/network settings

### Classification Always Falls Back

- Check server logs for errors
- Verify LLM configuration in server environment
- Test server directly with `mcp_classifier_client.py`

## Integration with Main App

The main Graph RAG application (`unified_search.py`) will automatically use MCP when enabled:

```python
# In unified_search.py
classification = await classify_question(question)
# This will use MCP if USE_MCP_CLASSIFIER=true
# Otherwise uses direct classification
```

No code changes needed in the main app - just configuration!

