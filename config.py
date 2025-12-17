"""
Centralized Configuration Module

This module loads, validates, and documents all configuration settings in one place,
reducing duplication and runtime surprises.

All configuration is loaded from environment variables with sensible defaults.
Validation ensures required settings are present and have valid values.
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger("Config")


@dataclass
class DatabaseConfig:
    """
    Database configuration settings for Neo4j connection.
    
    Attributes:
        url: Neo4j connection URL (e.g., "bolt://localhost:7687" or "bolt://neo4j:7687" for Docker)
        username: Neo4j username
        password: Neo4j password
        graph_name: Name of the graph projection (default: "entityGraph")
    """
    url: str
    username: str
    password: str
    graph_name: str = "entityGraph"
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load database config from environment"""
        url = os.getenv("DB_URL")
        username = os.getenv("DB_USERNAME")
        password = os.getenv("DB_PASSWORD")
        graph_name = os.getenv("GRAPH_NAME", "entityGraph")
        
        if not url:
            raise ValueError("DB_URL is required but not set in environment")
        if not username:
            raise ValueError("DB_USERNAME is required but not set in environment")
        if not password:
            raise ValueError("DB_PASSWORD is required but not set in environment")
        
        return cls(url=url, username=username, password=password, graph_name=graph_name)


@dataclass
class LLMConfig:
    """
    LLM configuration settings for answer generation and general LLM tasks.
    
    Attributes:
        provider: LLM provider ("google", "openai", "ollama")
        api_key: API key for LLM service (optional for some providers)
        model: Model name (e.g., "gemini-2.5-flash", "gpt-4", "llama3.2")
        base_url: Base URL for LLM API endpoint
        temperature: Sampling temperature (0.0 for deterministic, higher for creativity)
        timeout: Request timeout in seconds
        stop: Optional list of stop sequences
    """
    provider: str  # "google", "openai", "ollama"
    api_key: Optional[str]
    model: str
    base_url: str
    temperature: float
    timeout: int
    stop: Optional[List[str]] = None
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load LLM config from environment"""
        provider = os.getenv("LLM_PROVIDER", "google").lower()
        api_key_raw = os.getenv("OPENAI_API_KEY")
        api_key = api_key_raw.strip('"').strip("'") if api_key_raw else None
        model = os.getenv("OPENAI_MODEL", "gemini-2.5-flash")
        base_url = os.getenv("OPENAI_BASE_URL")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
        timeout = int(os.getenv("API_TIMEOUT", "600"))
        
        stop_str = os.getenv("OPENAI_STOP")
        stop = None
        if stop_str:
            try:
                import json
                stop = json.loads(stop_str) if stop_str else None
            except:
                stop = [stop_str]
        
        if not base_url:
            raise ValueError("OPENAI_BASE_URL is required but not set in environment")
        
        return cls(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            stop=stop
        )


@dataclass
class NERConfig:
    """Named Entity Recognition configuration"""
    model: str
    base_url: str
    api_key: Optional[str]
    temperature: float
    timeout: int
    
    @classmethod
    def from_env(cls) -> "NERConfig":
        """Load NER config from environment"""
        model = os.getenv("NER_MODEL", "gemini-2.5-flash")
        base_url = os.getenv("NER_BASE_URL")
        api_key_raw = os.getenv("NER_API_KEY", "test")
        api_key = api_key_raw.strip('"').strip("'") if api_key_raw else "test"
        temperature = float(os.getenv("NER_TEMPERATURE", "0.0"))
        timeout = int(os.getenv("API_TIMEOUT", "600"))
        
        if not base_url:
            raise ValueError("NER_BASE_URL is required but not set in environment")
        
        return cls(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            timeout=timeout
        )


@dataclass
class CorefConfig:
    """Coreference resolution configuration"""
    model: str
    base_url: str
    api_key: Optional[str]
    temperature: float
    timeout: int
    
    @classmethod
    def from_env(cls) -> "CorefConfig":
        """Load coreference config from environment"""
        model = os.getenv("COREF_MODEL", "gemini-2.5-flash")
        base_url = os.getenv("COREF_BASE_URL")
        api_key_raw = os.getenv("COREF_API_KEY", "test")
        api_key = api_key_raw.strip('"').strip("'") if api_key_raw else "test"
        temperature = float(os.getenv("COREF_TEMPERATURE", "0.0"))
        timeout = int(os.getenv("API_TIMEOUT", "600"))
        
        if not base_url:
            raise ValueError("COREF_BASE_URL is required but not set in environment")
        
        return cls(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            timeout=timeout
        )


@dataclass
class EmbeddingConfig:
    """Embedding service configuration"""
    api_url: str
    api_key: Optional[str]
    model_name: str
    dimension: int
    batch_size: int
    timeout: int
    provider: str
    include_bearer_auth: bool = True
    
    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Load embedding config from environment"""
        api_url = os.getenv("EMBEDDING_API_URL")
        api_key_raw = os.getenv("EMBEDDING_API_KEY")
        api_key = api_key_raw.strip('"').strip("'") if api_key_raw else None
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-001")
        dimension = int(os.getenv("EMBEDDING_DIMENSION", "768"))
        batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
        timeout = int(os.getenv("API_TIMEOUT", "600"))
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        include_bearer_auth = os.getenv("EMBEDDING_INCLUDE_BEARER_AUTH", "true").lower() == "true"
        
        if not api_url:
            raise ValueError("EMBEDDING_API_URL is required but not set in environment")
        
        return cls(
            api_url=api_url,
            api_key=api_key,
            model_name=model_name,
            dimension=dimension,
            batch_size=batch_size,
            timeout=timeout,
            provider=provider,
            include_bearer_auth=include_bearer_auth
        )


@dataclass
class MCPClassifierConfig:
    """MCP Classifier configuration"""
    enabled: bool
    url: str
    timeout: int
    
    @classmethod
    def from_env(cls) -> "MCPClassifierConfig":
        """Load MCP classifier config from environment"""
        enabled = os.getenv("USE_MCP_CLASSIFIER", "false").lower() == "true"
        url = os.getenv("MCP_CLASSIFIER_URL", "http://localhost:8001/mcp")
        timeout = int(os.getenv("MCP_CLASSIFIER_TIMEOUT", "30"))
        
        return cls(enabled=enabled, url=url, timeout=timeout)


@dataclass
class MCPNeo4jConfig:
    """
    MCP Neo4j Cypher server configuration for graph analytical queries.
    
    Attributes:
        enabled: Whether to use MCP Neo4j Cypher server for GRAPH_ANALYTICAL questions
        url: MCP Neo4j Cypher server URL (default: "http://localhost:8002/mcp")
        timeout: Request timeout in seconds
        max_refinement_iterations: Maximum number of query refinement iterations
    """
    enabled: bool
    url: str
    timeout: int
    max_refinement_iterations: int
    
    @classmethod
    def from_env(cls) -> "MCPNeo4jConfig":
        """Load MCP Neo4j config from environment"""
        enabled = os.getenv("USE_MCP_NEO4J", "true").lower() == "true"
        url = os.getenv("MCP_NEO4J_URL", "http://localhost:8002/mcp")
        timeout = int(os.getenv("MCP_NEO4J_TIMEOUT", "30"))
        max_refinement_iterations = int(os.getenv("MCP_NEO4J_MAX_REFINEMENT_ITERATIONS", "3"))
        
        return cls(
            enabled=enabled,
            url=url,
            timeout=timeout,
            max_refinement_iterations=max_refinement_iterations
        )


@dataclass
class ClassifierConfig:
    """Question classifier configuration"""
    use_heuristics: bool
    use_llm: bool
    mcp_config: MCPClassifierConfig
    
    @classmethod
    def from_env(cls) -> "ClassifierConfig":
        """Load classifier config from environment"""
        use_heuristics = os.getenv("CLASSIFIER_USE_HEURISTICS", "true").lower() == "true"
        use_llm = os.getenv("CLASSIFIER_USE_LLM", "true").lower() == "true"
        mcp_config = MCPClassifierConfig.from_env()
        
        return cls(
            use_heuristics=use_heuristics,
            use_llm=use_llm,
            mcp_config=mcp_config
        )


@dataclass
class MapReduceConfig:
    """Map-reduce configuration"""
    max_communities: int
    batch_size: int
    min_relevance: float
    
    @classmethod
    def from_env(cls) -> "MapReduceConfig":
        """Load map-reduce config from environment"""
        max_communities = int(os.getenv("MAP_REDUCE_MAX_COMMUNITIES", "50"))
        batch_size = int(os.getenv("MAP_REDUCE_BATCH_SIZE", "5"))
        min_relevance = float(os.getenv("MAP_REDUCE_MIN_RELEVANCE", "0.3"))
        
        return cls(
            max_communities=max_communities,
            batch_size=batch_size,
            min_relevance=min_relevance
        )


@dataclass
class ProcessingConfig:
    """Text processing configuration"""
    chunk_size: int
    chunk_overlap: int
    document_processing_batch_size: int
    batch_size: int
    max_workers: int
    cache_ttl: int
    cosine_epsilon: float
    
    @classmethod
    def from_env(cls) -> "ProcessingConfig":
        """Load processing config from environment"""
        chunk_size = int(os.getenv("CHUNK_SIZE_GDS", "512"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        document_processing_batch_size = int(os.getenv("DOCUMENT_PROCESSING_BATCH_SIZE", "20"))
        batch_size = int(os.getenv("BATCH_SIZE", "10"))
        max_workers = int(os.getenv("MAX_WORKERS", "4"))
        cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
        cosine_epsilon = float(os.getenv("COSINE_EPSILON", "1e-8"))
        
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            document_processing_batch_size=document_processing_batch_size,
            batch_size=batch_size,
            max_workers=max_workers,
            cache_ttl=cache_ttl,
            cosine_epsilon=cosine_epsilon
        )


@dataclass
class SearchConfig:
    """Search configuration"""
    relevance_threshold: float
    max_chunks_per_answer: int
    quick_search_max_chunks: int
    max_community_summaries: int
    similarity_threshold_chunks: float
    similarity_threshold_entities: float
    broad_search_max_communities: int
    
    @classmethod
    def from_env(cls) -> "SearchConfig":
        """Load search config from environment"""
        relevance_threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))
        max_chunks_per_answer = int(os.getenv("MAX_CHUNKS_PER_ANSWER", "7"))
        quick_search_max_chunks = int(os.getenv("QUICK_SEARCH_MAX_CHUNKS", "5"))
        max_community_summaries = int(os.getenv("MAX_COMMUNITY_SUMMARIES", "3"))
        similarity_threshold_chunks = float(os.getenv("SIMILARITY_THRESHOLD_CHUNKS", "0.4"))
        similarity_threshold_entities = float(os.getenv("SIMILARITY_THRESHOLD_ENTITIES", "0.6"))
        broad_search_max_communities = int(os.getenv("BROAD_SEARCH_MAX_COMMUNITIES", "20"))
        
        return cls(
            relevance_threshold=relevance_threshold,
            max_chunks_per_answer=max_chunks_per_answer,
            quick_search_max_chunks=quick_search_max_chunks,
            max_community_summaries=max_community_summaries,
            similarity_threshold_chunks=similarity_threshold_chunks,
            similarity_threshold_entities=similarity_threshold_entities,
            broad_search_max_communities=broad_search_max_communities
        )


@dataclass
class ExplainabilityConfig:
    """
    Explainability configuration for LLM responses with inline citations and references.
    
    Attributes:
        enabled: Feature flag to enable/disable explainability (inline citations)
        max_sources: Maximum sources to show in references block
        snippet_chars: Maximum characters per snippet in references
        only_cited_sources: Only show sources actually cited by the LLM (vs all retrieved sources)
    """
    enabled: bool                    # Feature flag to enable/disable explainability
    max_sources: int                 # Maximum sources to show in references block
    snippet_chars: int               # Maximum characters per snippet in references
    only_cited_sources: bool         # Only show sources actually cited by the LLM
    
    @classmethod
    def from_env(cls) -> "ExplainabilityConfig":
        """Load explainability config from environment"""
        enabled = os.getenv("EXPLAIN_ENABLED", "true").lower() == "true"
        max_sources = int(os.getenv("EXPLAIN_MAX_SOURCES", "8"))
        snippet_chars = int(os.getenv("EXPLAIN_SNIPPET_CHARS", "320"))
        only_cited_sources = os.getenv("EXPLAIN_ONLY_CITED_SOURCES", "true").lower() == "true"
        
        return cls(
            enabled=enabled,
            max_sources=max_sources,
            snippet_chars=snippet_chars,
            only_cited_sources=only_cited_sources
        )


@dataclass
class ResilienceConfig:
    """Resilience configuration for external service calls"""
    max_retries: int
    retry_backoff_factor: float
    retry_initial_delay: float
    circuit_breaker_failure_threshold: int
    circuit_breaker_success_threshold: int
    circuit_breaker_timeout: float
    request_timeout: float
    
    @classmethod
    def from_env(cls) -> "ResilienceConfig":
        """Load resilience config from environment"""
        max_retries = int(os.getenv("RESILIENCE_MAX_RETRIES", "3"))
        retry_backoff_factor = float(os.getenv("RESILIENCE_BACKOFF_FACTOR", "2.0"))
        retry_initial_delay = float(os.getenv("RESILIENCE_INITIAL_DELAY", "1.0"))
        circuit_breaker_failure_threshold = int(os.getenv("RESILIENCE_CB_FAILURE_THRESHOLD", "5"))
        circuit_breaker_success_threshold = int(os.getenv("RESILIENCE_CB_SUCCESS_THRESHOLD", "2"))
        circuit_breaker_timeout = float(os.getenv("RESILIENCE_CB_TIMEOUT", "60.0"))
        request_timeout = float(os.getenv("RESILIENCE_REQUEST_TIMEOUT", "30.0"))
        
        return cls(
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
            retry_initial_delay=retry_initial_delay,
            circuit_breaker_failure_threshold=circuit_breaker_failure_threshold,
            circuit_breaker_success_threshold=circuit_breaker_success_threshold,
            circuit_breaker_timeout=circuit_breaker_timeout,
            request_timeout=request_timeout
        )


@dataclass
class AppConfig:
    """Application configuration"""
    title: str
    description: str
    version: str
    host: str
    port: int
    log_level: str
    enable_cors: bool
    reload: bool
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load app config from environment"""
        title = os.getenv("APP_TITLE", "Graph RAG API")
        description = os.getenv("APP_DESCRIPTION", "Production-ready Graph RAG API")
        version = os.getenv("APP_VERSION", "2.0.0")
        host = os.getenv("APP_HOST", "0.0.0.0")
        port = int(os.getenv("APP_PORT", "8000"))
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        enable_cors = os.getenv("ENABLE_CORS", "true").lower() == "true"
        reload = os.getenv("RELOAD", "false").lower() == "true"
        
        return cls(
            title=title,
            description=description,
            version=version,
            host=host,
            port=port,
            log_level=log_level,
            enable_cors=enable_cors,
            reload=reload
        )


@dataclass
class Config:
    """Main configuration container"""
    app: AppConfig
    database: DatabaseConfig
    llm: LLMConfig
    ner: NERConfig
    coref: CorefConfig
    embedding: EmbeddingConfig
    classifier: ClassifierConfig
    map_reduce: MapReduceConfig
    processing: ProcessingConfig
    search: SearchConfig
    resilience: ResilienceConfig
    mcp_neo4j: MCPNeo4jConfig
    explainability: ExplainabilityConfig
    
    @classmethod
    def load(cls) -> "Config":
        """Load all configuration from environment"""
        try:
            return cls(
                app=AppConfig.from_env(),
                database=DatabaseConfig.from_env(),
                llm=LLMConfig.from_env(),
                ner=NERConfig.from_env(),
                coref=CorefConfig.from_env(),
                embedding=EmbeddingConfig.from_env(),
                classifier=ClassifierConfig.from_env(),
                map_reduce=MapReduceConfig.from_env(),
                processing=ProcessingConfig.from_env(),
                search=SearchConfig.from_env(),
                resilience=ResilienceConfig.from_env(),
                mcp_neo4j=MCPNeo4jConfig.from_env(),
                explainability=ExplainabilityConfig.from_env()
            )
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate URLs
        for name, url in [
            ("LLM base URL", self.llm.base_url),
            ("NER base URL", self.ner.base_url),
            ("Coref base URL", self.coref.base_url),
            ("Embedding API URL", self.embedding.api_url),
        ]:
            if url and not (url.startswith("http://") or url.startswith("https://")):
                issues.append(f"{name} must be a valid HTTP/HTTPS URL")
        
        # Validate numeric ranges
        if not 0.0 <= self.llm.temperature <= 2.0:
            issues.append("LLM temperature must be between 0.0 and 2.0")
        if not 0.0 <= self.search.relevance_threshold <= 1.0:
            issues.append("Relevance threshold must be between 0.0 and 1.0")
        if self.resilience.max_retries < 0:
            issues.append("Max retries must be non-negative")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (excluding sensitive data)"""
        return {
            "app": {
                "title": self.app.title,
                "version": self.app.version,
                "host": self.app.host,
                "port": self.app.port,
                "log_level": self.app.log_level
            },
            "database": {
                "url": self.database.url,
                "username": self.database.username,
                "graph_name": self.database.graph_name
            },
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "base_url": self.llm.base_url,
                "temperature": self.llm.temperature
            },
            "embedding": {
                "model_name": self.embedding.model_name,
                "dimension": self.embedding.dimension,
                "batch_size": self.embedding.batch_size
            },
            "processing": {
                "chunk_size": self.processing.chunk_size,
                "chunk_overlap": self.processing.chunk_overlap
            },
            "search": {
                "relevance_threshold": self.search.relevance_threshold,
                "max_chunks_per_answer": self.search.max_chunks_per_answer
            }
        }


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance (lazy loading)"""
    global _config
    if _config is None:
        _config = Config.load()
        issues = _config.validate()
        if issues:
            logger.warning(f"Configuration validation issues: {issues}")
    return _config


def reload_config() -> Config:
    """Reload configuration from environment"""
    global _config
    _config = Config.load()
    issues = _config.validate()
    if issues:
        logger.warning(f"Configuration validation issues: {issues}")
    return _config

