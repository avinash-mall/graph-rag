# MCP Classifier Server Docker Setup

## Overview

The MCP (Model Context Protocol) Classifier Server is now configured to run as a Docker service in `docker-compose.yaml`. This allows the MCP server to run in a container alongside Neo4j for easier deployment and management.

## Files Added/Modified

### New Files
- ✅ `Dockerfile.mcp` - Dockerfile for building the MCP classifier server image
- ✅ `DOCKER_MCP_SETUP.md` - This documentation file

### Modified Files
- ✅ `docker-compose.yaml` - Added MCP classifier server service
- ✅ `.env` - Added comments about Docker networking

## Docker Compose Configuration

The `docker-compose.yaml` now includes:

### Services

1. **neo4j** - Neo4j graph database (existing)
2. **mcp-classifier** - MCP Classifier Server (new)

### MCP Server Service Details

```yaml
mcp-classifier:
  build:
    context: .
    dockerfile: Dockerfile.mcp
  container_name: mcp-classifier-server
  ports:
    - "8001:8001"  # Exposed to host
  environment:
    - MCP_CLASSIFIER_PORT=8001
    - PORT=8001
    - CLASSIFIER_USE_HEURISTICS=true
    - CLASSIFIER_USE_LLM=true
    - LLM_PROVIDER=openai
    # ... other env vars
  env_file:
    - .env
  volumes:
    - .:/app  # Mount code for development
    - /app/__pycache__  # Exclude cache
  networks:
    - graph-rag-network
  healthcheck:
    # Health check every 30s
```

## Usage

### Starting Services

```bash
# Start both Neo4j and MCP server
docker-compose up -d

# Start only MCP server
docker-compose up -d mcp-classifier

# View logs
docker-compose logs -f mcp-classifier

# Stop services
docker-compose down
```

### Building the Image

```bash
# Build MCP server image
docker-compose build mcp-classifier

# Rebuild without cache
docker-compose build --no-cache mcp-classifier
```

### Health Check

The MCP server includes a health check:
```bash
# Check service status
docker-compose ps mcp-classifier

# Manual health check
docker-compose exec mcp-classifier python -c "import httpx; httpx.get('http://localhost:8001/mcp', timeout=5)"
```

## Networking

### Service Access

- **From host machine**: `http://localhost:8001/mcp`
- **From Docker containers**: `http://mcp-classifier-server:8001/mcp`

### Network Configuration

Both services are on the `graph-rag-network` bridge network:
- Services can communicate using service names
- Neo4j: accessible via `${NEO4J_CONTAINER_NAME}`
- MCP Server: accessible via `mcp-classifier-server`

## Configuration

### Environment Variables

The MCP server requires these environment variables (from `.env`):

- `MCP_CLASSIFIER_PORT` - Server port (default: 8001)
- `CLASSIFIER_USE_HEURISTICS` - Enable heuristics (default: true)
- `CLASSIFIER_USE_LLM` - Enable LLM classification (default: true)
- `LLM_PROVIDER` - LLM provider (openai, google)
- `OPENAI_API_KEY` - API key for LLM
- `OPENAI_BASE_URL` - LLM API base URL
- `OPENAI_MODEL` - Model name
- `LOG_LEVEL` - Logging level (default: INFO)

### Updating MCP URL for Docker

If your main application also runs in Docker, update `.env`:

```env
# For Docker-to-Docker communication
MCP_CLASSIFIER_URL=http://mcp-classifier-server:8001/mcp

# For host-to-Docker communication (main app runs locally)
MCP_CLASSIFIER_URL=http://localhost:8001/mcp
```

## Dockerfile Details

The `Dockerfile.mcp`:

- Uses Python 3.11-slim base image
- Installs system dependencies (gcc, g++)
- Copies and installs requirements.txt
- Copies all Python files
- Exposes port 8001
- Includes health check
- Runs `mcp_classifier_server.py`

## Development Workflow

### Hot Reload

Code changes are automatically reflected because:
- The code directory is mounted as a volume
- Just restart the container to pick up changes:
  ```bash
  docker-compose restart mcp-classifier
  ```

### Testing Changes

1. Make code changes
2. Restart container: `docker-compose restart mcp-classifier`
3. Check logs: `docker-compose logs -f mcp-classifier`
4. Test endpoint: `curl http://localhost:8001/mcp`

## Troubleshooting

### Server Won't Start

1. **Check logs**:
   ```bash
   docker-compose logs mcp-classifier
   ```

2. **Verify environment variables**:
   ```bash
   docker-compose exec mcp-classifier env | grep MCP
   ```

3. **Check port availability**:
   ```bash
   # On Windows
   netstat -ano | findstr :8001
   ```

### Build Errors

1. **Clear Docker cache**:
   ```bash
   docker system prune -a
   docker-compose build --no-cache mcp-classifier
   ```

2. **Check Dockerfile syntax**:
   ```bash
   docker build -f Dockerfile.mcp -t test-mcp .
   ```

### Connection Issues

1. **Verify network**:
   ```bash
   docker network inspect graph-rag_graph-rag-network
   ```

2. **Test from container**:
   ```bash
   docker-compose exec mcp-classifier curl http://localhost:8001/mcp
   ```

3. **Check service name resolution**:
   ```bash
   docker-compose exec mcp-classifier ping mcp-classifier-server
   ```

## Production Considerations

For production deployments:

1. **Remove volume mounts** - Copy files into image instead
2. **Use secrets** - Don't pass API keys via environment variables
3. **Add resource limits**:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '0.5'
         memory: 512M
   ```
4. **Enable HTTPS** - Use reverse proxy (nginx, traefik)
5. **Multi-stage build** - Reduce image size

## Example Commands

```bash
# Start all services
docker-compose up -d

# View all services
docker-compose ps

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild and restart MCP server
docker-compose up -d --build mcp-classifier

# Execute command in container
docker-compose exec mcp-classifier python -c "print('Hello from MCP')"
```

## Verification

After starting the services:

1. **Check service status**:
   ```bash
   docker-compose ps
   ```

2. **Test MCP endpoint**:
   ```bash
   curl http://localhost:8001/mcp
   ```

3. **Check health**:
   ```bash
   docker inspect mcp-classifier-server | grep -A 10 Health
   ```

## Next Steps

1. ✅ Docker configuration complete
2. ✅ Health checks configured
3. ✅ Networking setup
4. ⏭️ Optional: Add to CI/CD pipeline
5. ⏭️ Optional: Set up monitoring/logging

