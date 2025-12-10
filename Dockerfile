# Dockerfile for Graph RAG API
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./

# Create logs directory
RUN mkdir -p /app/logs

# Expose the API port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"]

