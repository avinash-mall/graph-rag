# MCP Classification Implementation Summary

## What Was Implemented

We've added MCP (Model Context Protocol) support for question classification, allowing the classifier to run as a separate service with tool calling capabilities.

## Files Created

### 1. `mcp_classifier_server.py`
- **Purpose**: MCP server that exposes `classify_question` tool
- **Features**:
  - Uses FastMCP for easy server setup
  - Exposes async `classify_question` tool
  - Uses same classification logic (LLM + heuristics)
  - Runs on HTTP transport (default port 8001)

### 2. `mcp_classifier_client.py`
- **Purpose**: Client to call the MCP classification server
- **Features**:
  - Connects to MCP server via HTTP
  - Calls `classify_question` tool
  - Handles errors gracefully with fallback
  - Includes connection testing

### 3. `MCP_CLASSIFIER_SETUP.md`
- **Purpose**: Complete setup and usage guide
- **Contents**: Installation, configuration, testing, deployment

## Files Modified

### 1. `question_classifier.py`
- Added `USE_MCP_CLASSIFIER` configuration option
- Modified `classify()` method to check for MCP first
- Falls back to direct classification if MCP unavailable
- Backward compatible - works with or without MCP

### 2. `requirements.txt`
- Added `mcp[cli]>=1.0.0` dependency

### 3. `.env`
- Added MCP configuration options:
  - `USE_MCP_CLASSIFIER`
  - `MCP_CLASSIFIER_URL`
  - `MCP_CLASSIFIER_PORT`
  - `MCP_CLASSIFIER_TIMEOUT`

## How It Works

### Architecture Flow

```
┌─────────────────────────────────────┐
│   Graph RAG FastAPI Application     │
│                                     │
│   unified_search.py                 │
│   └─> question_classifier.classify()│
└──────────────┬──────────────────────┘
               │
               │ Check: USE_MCP_CLASSIFIER?
               │
        ┌──────┴──────┐
        │             │
    Yes │             │ No
        │             │
        ▼             ▼
┌──────────────┐  ┌──────────────────┐
│ MCP Client   │  │ Direct           │
│              │  │ Classification   │
│ mcp_classifier_│  │ (LLM + Heuristics)│
│ client.py    │  └──────────────────┘
└──────┬───────┘
       │
       │ HTTP/MCP Protocol
       │
       ▼
┌──────────────┐
│ MCP Server   │
│              │
│ mcp_classifier_│
│ server.py    │
│              │
│ Tool:        │
│ classify_question()│
└──────┬───────┘
       │
       │ Uses
       ▼
┌──────────────┐
│ LLM Service  │
│ (via utils)  │
└──────────────┘
```

### Classification Process

1. **Question received** in `unified_search.py`
2. **Check configuration**: Is `USE_MCP_CLASSIFIER=true`?
   - **If Yes**: Call MCP server via `mcp_classifier_client.py`
   - **If No**: Use direct classification in `question_classifier.py`
3. **MCP Server** (if used):
   - Receives tool call
   - Runs classification (heuristics + LLM)
   - Returns structured result
4. **Result** used to route search strategy:
   - BROAD → Community summaries + map-reduce
   - CHUNK → Chunk-level retrieval
   - OUT_OF_SCOPE → Polite fallback

## Usage

### Option 1: Direct Classification (Default)

```bash
# In .env
USE_MCP_CLASSIFIER=false
```

Classification runs directly in the main process - no separate server needed.

### Option 2: MCP Classification (Recommended for Production)

**Step 1**: Start MCP Server

```bash
# Terminal 1
python mcp_classifier_server.py
# Server starts on http://localhost:8001/mcp
```

**Step 2**: Enable MCP in Graph RAG

```bash
# In .env
USE_MCP_CLASSIFIER=true
MCP_CLASSIFIER_URL=http://localhost:8001/mcp
```

**Step 3**: Start Graph RAG App

```bash
# Terminal 2
python main.py
```

The app will automatically use the MCP server for classification.

## Testing

### Test MCP Server

```bash
python mcp_classifier_client.py
```

Expected output:
```
Testing MCP classifier connection...
MCP server connection successful. Available tools: ['classify_question']

Testing classification...
Question: Give me an overview of all policies
Classification: BROAD
Reason: Question asks for overview/broad understanding
Confidence: 0.85
```

### Test from Graph RAG

Just make a search request:

```bash
curl -X POST "http://localhost:8000/api/search/search" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Give me an overview of all policies",
    "scope": "hybrid"
  }'
```

Check logs to see which classification method was used.

## Benefits

1. ✅ **Separation of Concerns**: Classification is isolated service
2. ✅ **Scalability**: Can scale classifier independently
3. ✅ **Tool Calling**: Uses standard MCP protocol
4. ✅ **Resilience**: Falls back to direct classification if MCP unavailable
5. ✅ **No Breaking Changes**: Backward compatible, opt-in feature

## Configuration Summary

| Setting | Default | Description |
|---------|---------|-------------|
| `USE_MCP_CLASSIFIER` | `false` | Enable MCP classification |
| `MCP_CLASSIFIER_URL` | `http://localhost:8001/mcp` | MCP server URL |
| `MCP_CLASSIFIER_PORT` | `8001` | MCP server port |
| `MCP_CLASSIFIER_TIMEOUT` | `30` | Connection timeout (seconds) |

## Next Steps

1. **Install MCP SDK**: `pip install "mcp[cli]"`
2. **Test MCP Server**: `python mcp_classifier_client.py`
3. **Enable in Production**: Set `USE_MCP_CLASSIFIER=true` in `.env`
4. **Deploy Separately**: Run MCP server as separate service

## Troubleshooting

- **Can't connect to MCP**: Check if server is running on correct port
- **Classification fails**: Check MCP server logs and LLM configuration
- **Fallback occurs**: Check MCP server availability and network

See `MCP_CLASSIFIER_SETUP.md` for detailed troubleshooting.

