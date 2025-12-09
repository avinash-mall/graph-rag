# MCP Classifier Server Docker Setup

## Overview

The MCP (Model Context Protocol) Classifier Server is now available as a Docker service in `docker-compose.yaml`. This allows the MCP server to run in a container alongside Neo4j.

## Docker Configuration

### Services

The `docker-compose.yaml` file includes:

1. **neo4j** - Neo4j graph database
2. **mcp-classifier** - MCP Classifier Server (new)

### MCP Server Configuration

The MCP server runs as a separate Docker container with:

- **Container Name**: `mcp-classifier-server`
- **Port**: `8001` (configurable via `MCP_CLASSIFIER_PORT`)
- **Network**: Connected to `graph-rag-network`
- **Health Check**: Automatically checks server availability
- **Restart Policy**: Follows Neo4j restart policy (default: `always`)

## Usage

### Starting the MCP Server

```bash
# Start only the MCP server
docker-compose up -d mcp-classifier

# Start both Neo4j and MCP server
docker-compose up -d

# View logs
docker-compose logs -f mcp-classifier
```

### Building the MCP Server Image

```bash
# Build the MCP server image
docker-compose build mcp-classifier

# Or build without cache
docker-compose build --no-cache mcp-classifier
```

### Stopping the MCP Server

```bash
# Stop the MCP server
docker-compose stop mcp-classifier

# Stop all services
docker-compose down
```

## Configuration

### Environment Variables

The MCP server uses environment variables from `.env` file:

- `MCP_CLASSIFIER_PORT` - Port for the MCP server (default: 8001)
- `CLASSIFIER_USE_HEURISTICS` - Enable heuristic classification (default: true)
- `CLASSIFIER_USE_LLM` - Enable LLM-based classification (default: true)
- `LLM_PROVIDER` - LLM provider (openai, google)
- `OPENAI_API_KEY` - API key for LLM
- `OPENAI_BASE_URL` - Base URL for LLM API
- `OPENAI_MODEL` - Model name
- `LOG_LEVEL` - Logging level (default: INFO)

### Network Access

When running in Docker:
- **From host**: Use `http://localhost:8001/mcp`
- **From other containers**: Use `http://mcp-classifier-server:8001/mcp`

### Updating Configuration for Docker

If the main application runs in Docker, update `MCP_CLASSIFIER_URL` in `.env`:

```env
# For Docker-to-Docker communication
MCP_CLASSIFIER_URL=http://mcp-classifier-server:8001/mcp

# For host-to-Docker communication (main app runs locally)
MCP_CLASSIFIER_URL=http://localhost:8001/mcp
```

## Health Check

The MCP server includes a health check that:
- Checks every 30 seconds
- Times out after 10 seconds
- Retries 3 times
- Waits 10 seconds before first check

View health status:
```bash
docker-compose ps mcp-classifier
```

## Troubleshooting

### Server Not Starting

1. Check logs:
   ```bash
   docker-compose logs mcp-classifier
   ```

2. Verify environment variables:
   ```bash
   docker-compose exec mcp-classifier env | grep MCP
   ```

3. Test the endpoint manually:
   ```bash
   curl http://localhost:8001/mcp
   ```

### Connection Issues

1. **From host machine**: Ensure port 8001 is exposed and accessible
2. **From other containers**: Use service name `mcp-classifier-server` instead of `localhost`
3. **Network**: Ensure both containers are on the same Docker network (`graph-rag-network`)

### Build Issues

If the Docker image fails to build:

1. Check Python version compatibility
2. Verify all dependencies in `requirements.txt` are valid
3. Check Dockerfile syntax:
   ```bash
   docker build -f Dockerfile.mcp -t mcp-classifier-test .
   ```

## File Structure

The MCP server container includes:
- `mcp_classifier_server.py` - Main server file
- `utils.py` - Utility functions and LLM client
- `question_classifier.py` - Question classification logic
- All other Python files in the project directory

## Development

For development with hot-reload:

The docker-compose configuration mounts the current directory as a volume, so code changes are reflected immediately. Just restart the container:

```bash
docker-compose restart mcp-classifier
```

## Production Considerations

For production:

1. **Remove volume mounts** - Use COPY in Dockerfile instead
2. **Use multi-stage builds** - Reduce image size
3. **Add resource limits** - Set memory and CPU limits
4. **Use secrets** - Don't pass API keys via environment variables
5. **Enable SSL/TLS** - Use HTTPS for external access

## Example Docker Compose Command

```bash
# Start all services
docker-compose up -d

# Start only Neo4j
docker-compose up -d neo4j

# Start only MCP server
docker-compose up -d mcp-classifier

# View all services
docker-compose ps

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

