# Graph RAG API - Production Ready Implementation

## Overview

Graph RAG (Graph Retrieval-Augmented Generation) is a production-ready FastAPI application that combines Neo4j graph database with advanced LLM-based NLP processing and vector similarity search. The system processes documents (PDF, DOCX, TXT), builds knowledge graphs, and provides intelligent search capabilities with significant performance optimizations.

### Core Features

- **LLM-based NLP Processing** - Uses configurable LLM models (e.g., Gemini, OpenAI-compatible) for efficient entity extraction and coreference resolution
- **Intelligent Question Classification** - MCP-based routing that classifies questions as BROAD, CHUNK, GRAPH_ANALYTICAL, or OUT_OF_SCOPE for optimal search strategy
- **Map-Reduce for Broad Questions** - Processes community summaries using map-reduce pattern for comprehensive overview answers
- **Batch Embedding Optimization** - 10x speed improvement through intelligent batching and caching for chunks, entities, and search terms
- **Embedding-Based Entity Matching** - Vector similarity search for fuzzy entity matching in Cypher queries with high similarity thresholds
- **Unified Search Pipeline** - Single, flexible endpoint with intelligent routing based on question type
- **Vector Similarity Search** - Native Neo4j vector indexes for fast semantic search
- **Graph-Based Context Expansion** - Entity relationship traversal for comprehensive answers
- **Community Detection** - Leiden algorithm for automatic topic clustering and summarization
- **MMR Reranking** - Maximal Marginal Relevance for diverse, high-quality results
- **Comprehensive Async Handling** - Non-blocking operations throughout the pipeline
- **Production-Ready Architecture** - Centralized configuration, resilience patterns, and structured logging
- **Resilience Patterns** - Circuit breakers and automatic retries for all external service calls
- **Structured Logging** - Consistent, context-rich logging across all modules for better observability
- **Graph Analytical Queries** - MCP Neo4j Cypher integration for complex graph queries and multi-hop reasoning
- **Explainability** - Inline citations and source references with full text content for transparency

## Table of Contents

- [Overview](#overview)
- [Architecture &amp; Core Logic](#architecture--core-logic)
- [Document Processing Pipeline](#document-processing-pipeline)
- [Search Pipeline](#search-pipeline)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Achievements &amp; Current Status](#achievements--current-status)
- [Future Enhancements](#future-enhancements)
- [Testing](#testing)
- [Performance Optimizations](#performance-optimizations)

---

## Architecture & Core Logic

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Document API │  │  Search API  │  │ System API   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼─────────────┐
│                  Core Processing Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ NLP Processor│  │ Embedding    │  │ LLM Client   │      │
│  │ (LLM-based)  │  │ Client       │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Text         │  │ Graph        │  │ MCP Neo4j    │      │
│  │ Processor    │  │ Manager      │  │ Cypher Client│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ Question     │  │ Map-Reduce   │                        │
│  │ Classifier   │  │ Processor    │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────┬──────────────────┬───────────────────────────────┘
          │                  │
          │                  │
┌─────────▼──────────────────▼───────────────────────────────┐
│              Neo4j Graph Database                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Chunks      │  │  Entities    │  │ Communities  │      │
│  │  (with       │  │  (with       │  │  (with       │      │
│  │  vectors)    │  │  vectors)    │  │  summaries)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Relationships: MENTIONED_IN, RELATES_TO, CO_OCCURS│   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Main Application** (`main.py`)

   - FastAPI application with lifespan management
   - Structured request logging and error handling middleware
   - Health checks and monitoring endpoints
   - Uses centralized configuration and logging

2. **Configuration Management** (`config.py`)

   - **Centralized configuration module** - Single source of truth for all settings
   - Loads, validates, and documents all configuration from environment variables
   - Type-safe dataclasses for each configuration section
   - Reduces duplication and runtime surprises
   - Provides configuration validation on startup

3. **Resilience Patterns** (`resilience.py`)

   - **Circuit breaker implementation** - Prevents cascading failures (CLOSED/OPEN/HALF_OPEN states)
   - **Automatic retries with exponential backoff** - Handles transient failures gracefully
   - **Timeout handling** - Prevents hanging operations
   - **Per-service circuit breakers** - Separate breakers for LLM, embedding, NER, coref, Neo4j, MCP
   - All external service calls automatically wrapped with resilience patterns

4. **Structured Logging** (`logging_config.py`)

   - **Standardized logging** - Consistent format across all modules
   - **Structured context fields** - Request IDs, operation names, service context
   - **JSON-formatted logs** - Optional for production (enabled via LOG_JSON)
   - **Request tracking** - Context variables for request/operation correlation
   - **Helper functions** - Common logging patterns (function calls, external services, DB operations)

5. **Document Processing** (`document_api.py`)

   - Multi-format file processing (PDF, DOCX, TXT)
   - LLM-based entity extraction (configurable model, e.g., Gemini, OpenAI-compatible)
   - LLM-based coreference resolution
   - Batch embedding generation with caching
   - Graph construction with cross-document entity merging
   - Community detection (Leiden algorithm) and summarization
   - All external calls use resilience patterns and structured logging

6. **Unified Search** (`unified_search.py`)

   - Intelligent question classification and routing
   - Map-reduce processing for broad questions
   - Vector similarity search with native Neo4j indexes
   - Graph-aware context expansion
   - Graph-aware reranking with entity overlap
   - MMR diversity reranking
   - Community summary integration
   - Resilient external service calls with structured logging

7. **Question Classification** (`question_classifier.py`)

   - MCP-based classification (optional)
   - Heuristic-based classification (fast fallback)
   - LLM-based classification (accurate)
   - Routes questions to appropriate search strategies (BROAD/CHUNK/GRAPH_ANALYTICAL/OUT_OF_SCOPE)
   - Uses centralized config, resilience, and structured logging

8. **MCP Neo4j Cypher Client** (`mcp_neo4j_cypher.py`)

   - **Graph schema retrieval**: Gets Neo4j schema (labels, relationships, properties) via MCP or direct query
   - **Term extraction**: Extracts key terms from natural language questions
   - **Embedding-based fuzzy entity matching**: Finds entities matching search terms using vector similarity search with embeddings (high similarity threshold), with text-based fallback
   - **Multi-strategy query generation**: Generates 5 different Cypher query strategies:
     - Fuzzy entity search
     - Relationship exploration
     - Path finding (1-3 hops)
     - Related entities via RELATES_TO
     - Chunk context retrieval
   - **Parallel query execution**: Executes all strategies in parallel for maximum coverage
   - **LLM-based query generation**: Fallback to LLM-generated queries when structured queries fail
   - **Iterative query refinement**: Refines queries based on execution feedback (up to 3 iterations)
   - **Result combination**: Combines and deduplicates results from multiple queries
   - **Answer formatting**: Uses LLM to format graph query results into natural language
   - **Explainability integration**: Provides detailed query strategy and result information
   - Uses centralized config, resilience, and structured logging

9. **Map-Reduce Processing** (`map_reduce.py`)

   - Map step: Extract relevant info from community summaries
   - Reduce step: Synthesize partial answers into comprehensive response
   - Optimized for broad/overview questions
   - Uses resilience patterns for LLM calls

10. **Utilities** (`utils.py`)

   - LLM-based NLP processing (NER, coreference resolution)
   - Batch embedding client with caching (TTL-based)
   - Advanced text cleaning and chunking
   - Async wrappers for database operations with resilience
   - Support for both OpenAI-compatible and Gemini APIs
   - All external calls protected with circuit breakers and retries

---

## Document Processing Pipeline

The document processing pipeline transforms raw documents into a searchable knowledge graph.

### Processing Flow

```mermaid
graph TD
    A[Upload Document] --> B{File Type?}
    B -->|PDF| C[Extract Text with PyPDF]
    B -->|DOCX| D[Extract Text with python-docx]
    B -->|TXT| E[Read Text Directly]
  
    C --> F[Clean Text]
    D --> F
    E --> F
  
    F --> G[Remove Boilerplate/Navigation]
    G --> H[Coreference Resolution LLM]
    H --> I[Chunk Text with Overlap]
  
    I --> J[Batch Generate Embeddings]
    J --> K[Extract Entities LLM]
  
    K --> L[Create Chunk Nodes]
    K --> M[Create Entity Nodes]
    K --> N[Create MENTIONED_IN Relationships]
  
    L --> O[Create RELATES_TO Relationships]
    M --> O
    N --> O
  
    O --> P[Community Detection Leiden]
    P --> Q[Generate Community Summaries]
    Q --> R[Store in Neo4j]
  
    style A fill:#e1f5ff
    style R fill:#c8e6c9
    style J fill:#fff9c4
    style K fill:#fff9c4
    style P fill:#f3e5f5
```

### Key Processing Steps

#### 1. Text Extraction & Cleaning

- **PDF Processing**: Uses PyPDF for text extraction from PDF files
- **DOCX Processing**: Uses python-docx for Word document parsing
- **Text Cleaning**: Removes boilerplate, navigation text, headers/footers, URLs
- **Coreference Resolution**: LLM-based resolution using gemma3:1b model

#### 2. Text Chunking

- **Sentence-aware chunking**: Uses BlingFire for sentence boundary detection
- **Overlap management**: Configurable overlap (default 50 chars) to maintain context
- **Size optimization**: Configurable chunk size (default 512 tokens)

#### 3. Entity Extraction

- **LLM-based NER**: Uses configurable LLM model (default: Gemini, configurable via NER_MODEL, NER_BASE_URL)
- **Entity Types**: Supports 18+ entity types (PERSON, ORGANIZATION, LOCATION, etc.)
- **Cross-Document Merging**: Entities with same name are merged across documents
- **Batch Processing**: Processes multiple chunks efficiently
- **Entity Embeddings**: Batch-generated embeddings for all unique entities, stored in database for vector similarity search

#### 4. Graph Construction

```mermaid
graph LR
    A[Document] --> B[Chunks]
    B --> C[Entities]
    C --> D[MENTIONED_IN]
    C --> E[RELATES_TO]
    E --> F[Communities]
    F --> G[CommunitySummaries]
  
    style B fill:#e3f2fd
    style C fill:#f3e5f5
    style F fill:#fff9c4
    style G fill:#c8e6c9
```

**Graph Schema:**

- **Chunk Nodes**: Store text, embeddings, document metadata
- **Entity Nodes**: Store entity names, types, embeddings
- **MENTIONED_IN**: Relationships between entities and chunks
- **RELATES_TO**: Relationships between co-occurring entities
- **CommunitySummary Nodes**: Store community summaries with embeddings

#### 5. Community Detection

- **Leiden Algorithm**: Uses Neo4j GDS for community detection
- **Fallback Method**: Simple co-occurrence clustering if Leiden fails
- **Automatic Summarization**: LLM-generated summaries for each community

---

## Search Pipeline

The unified search pipeline provides intelligent, context-aware answers by combining question classification, intelligent routing, vector search, graph traversal, Cypher queries, and advanced reranking. The system automatically routes questions to the most appropriate search strategy based on question type.

### Search Flow

```mermaid
graph TD
    A[User Query] --> B[Question Classification]
    B --> C{Question Type?}
    C -->|BROAD| D[Map-Reduce with Communities]
    C -->|CHUNK| E[Chunk-Level Vector Search]
    C -->|GRAPH_ANALYTICAL| CY[Cypher Query Processing]
    C -->|OUT_OF_SCOPE| F[Return Polite Fallback]
  
    D --> D1[Retrieve Community Summaries]
    D1 --> D2[Map: Extract from Each Summary]
    D2 --> D3[Reduce: Synthesize Final Answer]
    D3 --> X
  
    E --> G{Has Conversation History?}
    G -->|Yes| H[Rewrite Query with Context]
    G -->|No| I[Use Query as-is]
    H --> J[Generate Query Embedding]
    I --> J
  
    J --> K{Search Scope?}
    K -->|Global| L[Vector Search All Documents]
    K -->|Local| M[Vector Search Specific Doc]
    K -->|Hybrid| N[Combine Global + Local]
  
    L --> O[Retrieve Top-K Chunks]
    M --> O
    N --> O
  
    O --> P{Graph Expansion?}
    P -->|Yes| Q[Expand via Entity Relationships]
    P -->|No| R[Skip Expansion]
    Q --> S[Graph-Aware Reranking]
    R --> S
  
    S --> T[Entity Overlap Scoring]
    T --> U[Community Relevance Scoring]
    U --> V[Centrality Scoring]
    V --> W[MMR Diversity Reranking]
  
    W --> Y[Build Context String]
    Y --> Z[Generate Answer with LLM]
    Z --> AA[Calculate Confidence Score]
    AA --> X[Return Response]
  
    CY --> CY1[Extract Key Terms from Question]
    CY1 --> CY2[Generate Term Embeddings]
    CY2 --> CY3[Vector Search Entities]
    CY3 --> CY4{High Similarity Matches?}
    CY4 -->|Yes| CY5[Use Vector Matches]
    CY4 -->|No| CY6[Fallback to Text Matching]
    CY6 --> CY5
    CY5 --> CY7[Get Neo4j Graph Schema]
    CY3 --> CY4[Generate Multiple Query Strategies]
    CY4 --> CY5[Execute All Queries in Parallel]
    CY5 --> CY6{Results Found?}
    CY6 -->|Yes| CY7[Combine & Deduplicate Results]
    CY6 -->|No| CY8[Try LLM-Generated Query]
    CY8 --> CY7
    CY7 --> CY8[Generate Multiple Query Strategies]
    CY8 --> CY9[Execute All Queries in Parallel]
    CY9 --> CY10{Results Found?}
    CY10 -->|Yes| CY11[Combine & Deduplicate Results]
    CY10 -->|No| CY12[Try LLM-Generated Query]
    CY12 --> CY11
    CY11 --> CY13[Format Answer with LLM]
    CY13 --> CY14[Add Explainability Block]
    CY14 --> X
  
    style A fill:#e1f5ff
    style B fill:#fff9c4
    style D fill:#f3e5f5
    style CY fill:#ffeb3b
    style CY1 fill:#ffeb3b
    style CY2 fill:#ffeb3b
    style CY3 fill:#ffeb3b
    style CY4 fill:#ffeb3b
    style CY5 fill:#ffeb3b
    style CY6 fill:#fff9c4
    style CY7 fill:#ffeb3b
    style CY8 fill:#ffeb3b
    style CY9 fill:#ffeb3b
    style CY10 fill:#ffeb3b
    style CY11 fill:#ffeb3b
    style CY12 fill:#ffeb3b
    style CY13 fill:#ffeb3b
    style CY14 fill:#ffeb3b
    style Z fill:#fff9c4
    style X fill:#c8e6c9
    style W fill:#f3e5f5
```

### Search Components

#### 1. Question Classification

The system uses intelligent question classification to route queries to the optimal search strategy:

- **MCP-based Classification** (if enabled): Uses Model Context Protocol server for consistent, accurate classification
- **Heuristic Classification**: Fast keyword-based fallback using pattern matching
- **LLM Classification**: Direct LLM-based classification for accuracy when MCP unavailable

**Question Types:**

- **BROAD**: Questions requiring overview/understanding (e.g., "What are the main topics?") → Uses map-reduce with community summaries
- **CHUNK**: Questions requiring specific details (e.g., "What is the deadline?") → Uses chunk-level vector similarity search
- **GRAPH_ANALYTICAL**: Questions requiring graph analysis, aggregations, or multi-hop reasoning (e.g., "How are X and Y related?", "Count all entities", "Find paths between entities") → Uses MCP Neo4j Cypher queries with multi-strategy approach
- **OUT_OF_SCOPE**: Questions not answerable from knowledge base → Returns polite fallback

#### 2. Map-Reduce for Broad Questions

For BROAD questions, the system uses a map-reduce pattern:

- **Map Step**: Extract relevant information from each community summary using LLM
- **Reduce Step**: Synthesize partial answers into a comprehensive final answer using LLM
- **Batch Processing**: Processes multiple community summaries in parallel batches
- **Relevance Filtering**: Only includes summaries above minimum relevance threshold

#### 3. Query Preprocessing

```python
# Query rewriting with conversation history
if conversation_history:
    rewritten_query = llm.rewrite(query, conversation_history)
else:
    rewritten_query = query
```

#### 4. Vector Similarity Search

- **Native Neo4j Vector Index**: Uses `db.index.vector.queryNodes()` for fast search (Neo4j 5.11+)
- **Fallback to GDS**: Uses `gds.similarity.cosine()` if vector indexes unavailable
- **Threshold Filtering**: Configurable similarity threshold (default 0.4)
- **Multi-Index Support**: Separate indexes for Chunks, Entities, and CommunitySummaries

#### 5. Graph-Based Context Expansion

```mermaid
graph LR
    A[Initial Chunks] --> B[Extract Entities]
    B --> C[Find Related Entities]
    C --> D[Traverse RELATES_TO]
    D --> E[Find Related Chunks]
    E --> F[Add to Context]
  
    style A fill:#e3f2fd
    style F fill:#c8e6c9
```

#### 6. Advanced Reranking

**Graph-Aware Reranking:**

- **Entity Overlap Score**: Measures overlap between query entities and chunk entities
- **Community Relevance**: Similarity to relevant community summaries
- **Centrality Score**: Graph centrality of chunks (degree centrality)

**MMR Reranking:**

- **Maximal Marginal Relevance**: Balances relevance and diversity
- **Formula**: `MMR(doc) = λ × relevance(doc, query) - (1-λ) × max_sim(doc, selected)`
- **Lambda Parameter**: Default 0.7 (favors relevance over diversity)

#### 7. Cypher Query Processing (GRAPH_ANALYTICAL Questions)

For GRAPH_ANALYTICAL questions, the system uses a sophisticated multi-strategy Cypher query approach via the MCP Neo4j Cypher client. This enables complex graph analysis, aggregations, path finding, and relationship exploration that goes beyond simple vector similarity search.

**Process Flow:**

1. **Term Extraction**: Extracts key terms from the natural language question
2. **Embedding-Based Entity Matching**: 
   - Generates embeddings for search terms
   - Uses vector similarity search via Neo4j vector index (`entity_embedding_index`) to find entities with high semantic similarity
   - Filters results by configurable similarity threshold (default 0.6) to ensure high-quality matches
   - Falls back to text-based fuzzy matching (CONTAINS, toLower) if vector search fails or returns no results
3. **Schema Retrieval**: Gets the Neo4j graph schema (node labels, relationship types, properties) via MCP or direct query
4. **Multi-Strategy Query Generation**: Generates 5 different query strategies:
   - **Vector-Based Entity Search**: Find entities matching search terms using vector similarity (high similarity threshold)
   - **Relationship Exploration**: Find connections between matched entities
   - **Path Finding**: Find paths connecting entities (1-3 hops)
   - **Related Entities**: Find entities related via RELATES_TO relationships
   - **Chunk Context**: Find document chunks mentioning the entities
5. **Parallel Execution**: Executes all query strategies in parallel
6. **Result Combination**: Combines and deduplicates results from all successful queries
7. **LLM Fallback**: If no results, generates a custom Cypher query using LLM with schema context
8. **Answer Formatting**: Uses LLM to format query results into natural language answers
9. **Explainability**: Adds detailed explainability block showing:
   - Extracted terms and fuzzy matches
   - Query strategies attempted and their results
   - Best query used
   - Sample results

**Query Patterns Used:**

```cypher
# Fuzzy entity search
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower('term')
RETURN e.name, e.type LIMIT 10

# Relationship exploration
MATCH (e1:Entity)-[r]-(e2:Entity)
WHERE toLower(e1.name) CONTAINS toLower('term1')
  AND toLower(e2.name) CONTAINS toLower('term2')
RETURN e1.name, type(r), e2.name LIMIT 20

# Path finding
MATCH path = (e1:Entity)-[*1..3]-(e2:Entity)
WHERE toLower(e1.name) CONTAINS toLower('term1')
  AND toLower(e2.name) CONTAINS toLower('term2')
RETURN path LIMIT 10

# Related entities
MATCH (e1:Entity)-[:RELATES_TO]-(related:Entity)
WHERE toLower(e1.name) CONTAINS toLower('term')
RETURN e1.name, related.name, related.type LIMIT 20
```

**Key Features:**
- **Embedding-Based Vector Search**: Uses semantic similarity for entity matching, providing better accuracy than text-based matching
- **High Similarity Threshold**: Configurable threshold (default 0.6) ensures only high-quality matches are returned
- **Intelligent Fallback**: Automatically falls back to text-based fuzzy matching if vector search fails or returns no results
- **Iterative Refinement**: Can refine queries based on execution feedback (up to 3 iterations)
- **Multi-Strategy Approach**: Tries multiple query patterns to maximize result coverage
- **MCP Integration**: Uses MCP Neo4j Cypher server for query execution (with direct Neo4j fallback)
- **Comprehensive Answers**: Combines results from multiple strategies into coherent answers

#### 8. Answer Generation

- **Context Building**: Combines community summaries and relevant chunks (for BROAD/CHUNK questions)
- **Cypher Results Formatting**: Formats graph query results into natural language using LLM (for GRAPH_ANALYTICAL questions)
- **LLM Prompting**: Structured prompts with context and conversation history
- **Confidence Scoring**: Based on chunk similarity, summary availability, query results, and quality
- **Explainability**: Adds inline citations and source references when enabled
  - For Cypher queries: Shows extracted terms, fuzzy matches, query strategies, and sample results
  - For chunk-based queries: Shows source chunks with full text and relevance scores

---

### Cypher Query Processing Details

The Cypher query processing system (`mcp_neo4j_cypher.py`) provides a comprehensive solution for answering graph analytical questions. When a question is classified as GRAPH_ANALYTICAL, the system uses a multi-strategy approach to generate and execute Cypher queries.

#### Query Strategy Types

The system generates 5 different query strategies to maximize result coverage:

1. **Vector-Based Entity Search**
   - Generates embeddings for search terms
   - Uses Neo4j vector index (`entity_embedding_index`) for fast semantic similarity search
   - Filters by high similarity threshold (configurable, default 0.6) to ensure quality matches
   - Falls back to text-based fuzzy matching (CONTAINS, toLower) if vector search unavailable
   - Returns entity names, types, and similarity scores

2. **Relationship Exploration**
   - Finds direct relationships between matched entities
   - Explores MENTIONED_IN, RELATES_TO, and CO_OCCURS relationships
   - Returns relationship types and connected entities

3. **Path Finding**
   - Finds paths connecting entities (1-3 hops)
   - Uses variable-length path matching: `(e1)-[*1..3]-(e2)`
   - Returns path nodes and path length

4. **Related Entities**
   - Finds entities related via RELATES_TO relationships
   - Explores entity neighborhoods
   - Returns related entity names and types

5. **Chunk Context**
   - Finds document chunks mentioning the entities
   - Provides textual context for graph relationships
   - Returns chunk text and document names

#### Query Execution Flow

```mermaid
graph TD
    A[Natural Language Question] --> B[Extract Key Terms]
    B --> C[Generate Term Embeddings]
    C --> D[Vector Search Entities]
    D --> E{High Similarity Matches?}
    E -->|Yes| F[Use Vector Matches]
    E -->|No| G[Fallback: Text-Based Matching]
    G --> F
    F --> H[Get Graph Schema]
    H --> I[Generate 5 Query Strategies]
    I --> J[Execute All Queries in Parallel]
    J --> K{Any Results?}
    K -->|Yes| L[Combine & Deduplicate]
    K -->|No| M[LLM-Generated Query]
    M --> N[Execute LLM Query]
    N --> O{Results?}
    O -->|Yes| L
    O -->|No| P[Format No-Results Answer]
    L --> Q[Format Comprehensive Answer]
    Q --> R[Add Explainability Block]
    R --> S[Return Response]
    P --> S
    
    style A fill:#e1f5ff
    style C fill:#fff9c4
    style D fill:#ffeb3b
    style E fill:#fff9c4
    style F fill:#c8e6c9
    style G fill:#ffccbc
    style I fill:#f3e5f5
    style J fill:#ffeb3b
    style Q fill:#c8e6c9
    style R fill:#c8e6c9
```

#### Example Cypher Queries Generated

**Example 1: Vector-Based Entity Search**
```cypher
# First, vector search finds entities (handled internally via db.index.vector.queryNodes)
# Then uses matched entity names in queries:
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower('machine learning')
  OR e.name IN ['Machine Learning', 'ML', 'Artificial Intelligence']  # Vector-matched entities
RETURN e.name AS entity, e.type AS type
LIMIT 10
```

**Example 2: Relationship Exploration**
```cypher
MATCH (e1:Entity)-[r]-(e2:Entity)
WHERE toLower(e1.name) CONTAINS toLower('neural network')
  AND toLower(e2.name) CONTAINS toLower('deep learning')
RETURN e1.name AS from_entity, type(r) AS relationship, e2.name AS to_entity
LIMIT 20
```

**Example 3: Path Finding**
```cypher
MATCH (e1:Entity), (e2:Entity)
WHERE toLower(e1.name) CONTAINS toLower('AI')
  AND toLower(e2.name) CONTAINS toLower('robotics')
MATCH path = (e1)-[*1..3]-(e2)
RETURN [n IN nodes(path) | CASE WHEN 'Entity' IN labels(n) THEN n.name ELSE 'chunk' END] AS path_nodes,
       length(path) AS path_length
LIMIT 10
```

#### Explainability for Cypher Queries

When explainability is enabled (`EXPLAIN_ENABLED=true`), Cypher query responses include a detailed explainability block showing:

- **Extracted Terms**: Key terms identified from the question
- **Vector Matches**: How search terms were matched to entities using embedding-based vector search with similarity scores (e.g., "AI" → "Artificial Intelligence" with similarity 0.85)
- **Fallback Matches**: If vector search was unavailable, shows text-based fuzzy matches
- **Query Strategies**: All 5 strategies attempted with success/failure status (✓/✗) and result counts
- **Best Query**: The most successful Cypher query used (formatted in code block)
- **Sample Results**: First 5 results from the queries (truncated to snippet length)
- **Total Results**: Count of all results found across all strategies

This provides full transparency into how graph queries are processed and what data was retrieved, enabling users to understand and trust the answers provided.

---

## Quick Start

### Prerequisites

#### For Docker Compose Setup:
- **Docker** and **Docker Compose**
- **LLM Service**: Gemini API, OpenAI API, or compatible API (for NER and LLM)
- **Embedding Service**: Gemini, OpenAI, or compatible embedding API

#### For Local Development:
- **Python 3.11+**
- **Neo4j 5.0+** with GDS and APOC plugins (or use Docker for Neo4j)
- **LLM Service**: Gemini API, OpenAI API, or compatible API (for NER and LLM)
- **Embedding Service**: Gemini, OpenAI, or compatible embedding API

### Installation

#### Option 1: Docker Compose (Recommended)

The easiest way to run Graph RAG is using Docker Compose, which sets up all services (Neo4j, MCP Classifier, and Graph RAG API) automatically:

```bash
# 1. Clone the repository
git clone <repository-url>
cd graph-rag

# 2. Configure environment variables
cp .env.example .env
# Edit .env with your settings (API keys, model configurations, etc.)

# 3. Start all services
docker-compose up -d

# 4. Check service status
docker-compose ps

# 5. View logs (optional)
docker-compose logs -f graph-rag
```

The API will be available at `http://localhost:8000` with interactive documentation at `/docs`.

**Services:**
- **Graph RAG API**: http://localhost:8000
- **MCP Classifier Server**: http://localhost:8001
- **Neo4j Browser**: http://localhost:7474

#### Option 2: Local Development

For local development without Docker:

```bash
# 1. Clone the repository
git clone <repository-url>
cd graph-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Neo4j with Docker (if not running locally)
docker-compose up -d neo4j

# 5. Optionally start MCP classifier server
docker-compose up -d mcp-classifier

# 6. Configure environment variables
cp .env.example .env
# Edit .env with your settings

# 7. Run the application
python main.py
```

**Note**: For local development, update `DB_URL` in `.env` to `bolt://localhost:7687` instead of `bolt://neo4j:7687`.

---

## Configuration

### Centralized Configuration

All configuration is managed through the `config.py` module, which:
- Loads settings from environment variables with sensible defaults
- Validates configuration on startup
- Provides type-safe access to all settings
- Documents expected settings and their purposes

This eliminates configuration duplication and ensures consistent settings across all modules.

### Environment Variables

#### Application Settings

```bash
APP_TITLE="Graph RAG API"
APP_DESCRIPTION="Production-ready Graph RAG API"
APP_VERSION="2.0.0"
APP_HOST="0.0.0.0"
APP_PORT="8000"
ENABLE_CORS="true"
LOG_LEVEL="INFO"
LOG_JSON="false"  # Set to "true" for JSON-formatted logs in production
```

#### Database Configuration

**For Docker Compose setup:**
```bash
DB_URL="bolt://neo4j:7687"  # Use service name 'neo4j' when running in Docker
DB_USERNAME="neo4j"
DB_PASSWORD="neo4j123"
GRAPH_NAME="entityGraph"
```

**For local development (Neo4j running locally or via Docker):**
```bash
DB_URL="bolt://localhost:7687"  # Use 'localhost' when running locally
DB_USERNAME="neo4j"
DB_PASSWORD="neo4j123"
GRAPH_NAME="entityGraph"
```

**Note**: When using Docker Compose, the `DB_URL` in `.env` is automatically overridden to use the service name. For local development, ensure Neo4j is accessible at `localhost:7687`.

#### LLM Configuration

```bash
# Main LLM for answer generation
LLM_PROVIDER="google"  # or "openai"
OPENAI_API_KEY="your-api-key"
OPENAI_MODEL="gemini-2.5-flash"  # or "llama3.2", etc.
OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta"  # or "http://localhost:11434/v1" for Ollama
OPENAI_TEMPERATURE="0.0"

# NER Model
NER_MODEL="gemini-2.5-flash"  # or your preferred model
NER_BASE_URL="https://generativelanguage.googleapis.com/v1beta"  # or OpenAI-compatible URL
NER_API_KEY="your-api-key"
NER_TEMPERATURE="0.0"

# Coreference Resolution Model
COREF_MODEL="gemini-2.5-flash"  # or your preferred model
COREF_BASE_URL="https://generativelanguage.googleapis.com/v1beta"  # or OpenAI-compatible URL
COREF_API_KEY="your-api-key"
COREF_TEMPERATURE="0.0"
```

#### Embedding Configuration

```bash
# Embedding API URL (format depends on provider)
# Gemini: Base URL without /embeddings (e.g., https://generativelanguage.googleapis.com/v1beta)
# OpenAI/Ollama: Include /embeddings (e.g., http://localhost:11434/v1/embeddings)
EMBEDDING_API_URL="https://generativelanguage.googleapis.com/v1beta"
EMBEDDING_MODEL_NAME="gemini-embedding-001"  # or "mxbai-embed-large", "text-embedding-ada-002", etc.
EMBEDDING_API_KEY="your-api-key"
EMBEDDING_DIMENSION="768"  # Must match your embedding model dimension
EMBEDDING_BATCH_SIZE="10"
```

#### Question Classifier Configuration

```bash
# Enable heuristic-based classification (keyword matching)
CLASSIFIER_USE_HEURISTICS="true"

# Enable LLM-based classification (more accurate)
CLASSIFIER_USE_LLM="true"

# Use MCP (Model Context Protocol) classifier server
USE_MCP_CLASSIFIER="true"

# MCP Classifier Server URL
MCP_CLASSIFIER_URL="http://localhost:8001/mcp"
MCP_CLASSIFIER_PORT="8001"
MCP_CLASSIFIER_TIMEOUT="30"
```

#### Map-Reduce Configuration

```bash
# Maximum communities to process in map-reduce
MAP_REDUCE_MAX_COMMUNITIES="50"
MAP_REDUCE_BATCH_SIZE="5"
MAP_REDUCE_MIN_RELEVANCE="0.3"
```

#### MCP Neo4j Configuration

```bash
# Enable MCP Neo4j Cypher server for graph analytical queries
USE_MCP_NEO4J="true"
MCP_NEO4J_URL="http://localhost:8002/mcp"
MCP_NEO4J_TIMEOUT="30"
MCP_NEO4J_MAX_REFINEMENT_ITERATIONS="3"
```

#### Explainability Configuration

```bash
# Enable inline citations and source references
EXPLAIN_ENABLED="true"
EXPLAIN_MAX_SOURCES="8"
EXPLAIN_SNIPPET_CHARS="320"
EXPLAIN_ONLY_CITED_SOURCES="true"
```

#### Resilience Configuration

```bash
# Retry Configuration
RESILIENCE_MAX_RETRIES="3"  # Maximum number of retry attempts
RESILIENCE_BACKOFF_FACTOR="2.0"  # Exponential backoff multiplier
RESILIENCE_INITIAL_DELAY="1.0"  # Initial delay in seconds before first retry

# Circuit Breaker Configuration
RESILIENCE_CB_FAILURE_THRESHOLD="5"  # Failures before circuit opens
RESILIENCE_CB_SUCCESS_THRESHOLD="2"  # Successes needed to close circuit
RESILIENCE_CB_TIMEOUT="60.0"  # Seconds before circuit transitions from OPEN to HALF_OPEN

# Request Timeout
RESILIENCE_REQUEST_TIMEOUT="30.0"  # Timeout in seconds for individual requests
```

**Resilience Patterns:**
- All external service calls (LLM, embeddings, Neo4j) automatically use circuit breakers and retries
- Circuit breakers prevent cascading failures by stopping requests to failing services
- Automatic retries handle transient failures with exponential backoff
- Separate circuit breakers per service (llm, embedding, ner, coref, neo4j, mcp_classifier)

#### Performance Tuning

```bash
# Text Processing
CHUNK_SIZE_GDS="512"
CHUNK_OVERLAP="50"
DOCUMENT_PROCESSING_BATCH_SIZE="20"

# Search Parameters
RELEVANCE_THRESHOLD="0.40"
MAX_CHUNKS_PER_ANSWER="7"
QUICK_SEARCH_MAX_CHUNKS="5"
MAX_COMMUNITY_SUMMARIES="3"
SIMILARITY_THRESHOLD_CHUNKS="0.4"
SIMILARITY_THRESHOLD_ENTITIES="0.6"  # Minimum similarity for entity vector search (0.0-1.0, higher = stricter)
BROAD_SEARCH_MAX_COMMUNITIES="20"

# Caching
CACHE_TTL="3600"  # 1 hour
BATCH_SIZE="10"
MAX_WORKERS="4"
```

#### Logging Configuration

```bash
# Log Level
LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Structured Logging
LOG_JSON="false"  # Set to "true" for JSON-formatted logs (recommended for production)
LOG_TO_FILE="false"  # Set to "true" to write logs to file
```

**Structured Logging Features:**
- Consistent log format across all modules
- Context fields automatically included (request_id, operation, service, etc.)
- JSON formatting available for production log aggregation
- Request tracking for correlation across services

---

## API Endpoints

### Document Management

#### Upload Documents

```http
POST /api/documents/upload_documents
Content-Type: multipart/form-data

files: [file1.pdf, file2.docx, ...]
```

**Response:**

```json
{
  "message": "Successfully processed 2 documents",
  "results": [
    {
      "doc_id": "uuid-here",
      "document_name": "file1.pdf",
      "chunks_created": 45,
      "entities_extracted": 123,
      "processing_time": 12.34
    }
  ]
}
```

#### List Documents

```http
GET /api/documents/documents
```

#### Delete Document

```http
DELETE /api/documents/delete_document
Content-Type: application/json

{
  "doc_id": "uuid-here"
}
```

#### Get Community Summaries

```http
GET /api/documents/community_summaries?doc_id=uuid-here
```

#### Document Statistics

```http
GET /api/documents/document_stats
```

### Search Endpoints

#### Unified Search

```http
POST /api/search/search
Content-Type: application/json

{
  "question": "What are the main topics?",
  "conversation_history": "Previous context...",
  "doc_id": "optional-doc-id",
  "scope": "hybrid",  // "global" | "local" | "hybrid"
  "max_chunks": 7
}
```

**Note**: This endpoint automatically classifies questions using MCP (if enabled) or heuristic/LLM-based classification, then routes to:

- **BROAD questions**: Map-reduce processing with community summaries
- **CHUNK questions**: Chunk-level vector similarity search
- **GRAPH_ANALYTICAL questions**: MCP Neo4j Cypher queries for graph analysis and multi-hop reasoning
- **OUT_OF_SCOPE questions**: Polite fallback response

**Response (BROAD/CHUNK questions):**

```json
{
  "answer": "Detailed answer based on context...",
  "citations": [
    {
      "citation_id": 1,
      "source_type": "chunk",
      "source_id": "chunk_123",
      "source_title": "Document: example.pdf",
      "full_text": "Complete chunk text...",
      "relevance_score": 0.85
    }
  ],
  "confidence_score": 0.85,
  "chunks_used": 7,
  "entities_found": ["Entity1", "Entity2"],
  "search_time": 1.23,
  "metadata": {
    "scope": "hybrid",
    "community_summaries_used": 2,
    "chunks_retrieved": 14,
    "question_type": "CHUNK"
  }
}
```

**Response (GRAPH_ANALYTICAL questions):**

```json
{
  "answer": "Based on the graph analysis, Entity1 and Entity2 are connected via RELATES_TO relationship...",
  "citations": [
    {
      "citation_id": 1,
      "source_type": "cypher_result",
      "source_id": "query_1",
      "source_title": "Fuzzy Search Strategy",
      "full_text": "entity1: Entity1, entity2: Entity2, relationship: RELATES_TO",
      "relevance_score": 0.8
    }
  ],
  "confidence_score": 0.8,
  "chunks_used": 0,
  "entities_found": ["Entity1", "Entity2"],
  "search_time": 3.45,
  "metadata": {
    "question_type": "GRAPH_ANALYTICAL",
    "strategy": "multi_cypher_query",
    "iterations": 5,
    "results_count": 12,
    "extracted_terms": ["entity1", "entity2"],
    "fuzzy_matches": {
      "entity1": ["Entity1"],
      "entity2": ["Entity2"]
    },
    "query_strategies": [
      {"type": "fuzzy_search", "results": 5},
      {"type": "relationship_exploration", "results": 3},
      {"type": "path_finding", "results": 2},
      {"type": "related_entities", "results": 2}
    ],
    "final_query": "MATCH (e1:Entity)-[r]-(e2:Entity) WHERE...",
    "explainability_enabled": true
  }
}
```

#### Quick Search

```http
POST /api/search/quick_search
Content-Type: application/json

{
  "question": "Quick question",
  "doc_id": "optional-doc-id"
}
```

#### Search Suggestions

```http
GET /api/search/search_suggestions?doc_id=optional-doc-id
```

#### Search Analytics

```http
GET /api/search/search_analytics
```

#### Explain Search

```http
POST /api/search/explain_search
Content-Type: application/json

{
  "question": "Your question",
  "scope": "hybrid"
}
```

### System Endpoints

#### Health Check

```http
GET /health
```

#### API Information

```http
GET /api/info
```

#### Interactive Documentation

```http
GET /docs  # Swagger UI
GET /redoc  # ReDoc
```

---

## Achievements & Current Status

### ✅ Completed Features

#### Core Functionality

- ✅ **Multi-format Document Processing** - PDF, DOCX, TXT support
- ✅ **LLM-based Entity Extraction** - Using configurable models (Gemini, OpenAI-compatible)
- ✅ **Coreference Resolution** - LLM-based pronoun resolution
- ✅ **Cross-Document Entity Merging** - Entities with same name merged across documents
- ✅ **Advanced Text Cleaning** - Boilerplate and navigation text removal
- ✅ **Intelligent Chunking** - Sentence-aware with overlap
- ✅ **Batch Embedding Generation** - With TTL-based caching (for chunks, entities, and search terms)
- ✅ **Neo4j Graph Storage** - Complete graph schema implementation
- ✅ **Vector Indexing** - Native Neo4j vector indexes for fast search

#### Search Capabilities

- ✅ **Intelligent Question Classification** - MCP-based or heuristic/LLM routing (BROAD/CHUNK/GRAPH_ANALYTICAL/OUT_OF_SCOPE)
- ✅ **Map-Reduce for Broad Questions** - Comprehensive processing of community summaries
- ✅ **Unified Search Pipeline** - Single endpoint with intelligent routing
- ✅ **Vector Similarity Search** - Native Neo4j and GDS fallback
- ✅ **Graph-Based Expansion** - Entity relationship traversal
- ✅ **Graph-Aware Reranking** - Entity overlap, community, centrality scoring
- ✅ **MMR Diversity Reranking** - Balanced relevance and diversity
- ✅ **Community Summary Integration** - Automatic topic summaries
- ✅ **Conversation History Support** - Context-aware query rewriting
- ✅ **Graph Analytical Queries** - MCP Neo4j Cypher integration for complex graph queries with embedding-based entity matching
- ✅ **Explainability** - Inline citations and source references with full text content

#### Community Detection

- ✅ **Leiden Algorithm** - Neo4j GDS community detection
- ✅ **Fallback Clustering** - Simple co-occurrence method
- ✅ **Automatic Summarization** - LLM-generated community summaries

#### Production Features

- ✅ **Centralized Configuration** - Single source of truth (`config.py`) with validation
- ✅ **Resilience Patterns** - Circuit breakers and automatic retries for all external services
- ✅ **Structured Logging** - Consistent, context-rich logging across all modules (`logging_config.py`)
- ✅ **Comprehensive Error Handling** - Graceful degradation with proper error context
- ✅ **Async Processing** - Non-blocking operations throughout
- ✅ **Request Logging** - Detailed structured logging for monitoring and debugging
- ✅ **Health Checks** - System and component status
- ✅ **Type Safety** - Full type hints throughout
- ✅ **Modular Architecture** - Clean separation of concerns

#### Performance Optimizations

- ✅ **Batch Processing** - Embeddings, entities, queries
- ✅ **Intelligent Caching** - TTL-based embedding cache
- ✅ **Vector Indexes** - Fast similarity search
- ✅ **Connection Pooling** - Efficient database connections
- ✅ **Thread Pool Management** - CPU-bound task optimization

### 🔄 In Progress / Partial

- 🔄 **Streaming Responses** - Basic structure exists, needs enhancement
- 🔄 **Multi-language Support** - Framework ready, needs language-specific models
- 🔄 **Advanced Analytics** - Basic analytics exist, needs expansion

## Future Enhancements

### High Priority

1. **Security & Authentication**

   - JWT-based authentication
   - Role-based access control (RBAC)
   - API key management
   - Rate limiting per user/API key
2. **Scalability Improvements**

   - Redis for distributed caching
   - Vector database integration (Faiss/Pinecone)
   - Horizontal scaling support
   - Database sharding strategies
3. **Enhanced Search**

   - Hybrid search (keyword + vector)
   - Query expansion techniques
   - Faceted search
   - Search result caching
4. **Performance Monitoring**

   - APM integration (e.g., Prometheus)
   - Detailed performance metrics
   - Query performance analysis
   - Resource usage tracking

### Medium Priority

5. **Advanced Features**

   - Multi-language support expansion
   - Custom entity types and models
   - Graph visualization API
   - Document versioning
6. **Developer Experience**

   - SDK/Client libraries
   - Comprehensive tutorials
   - Example applications
   - Migration tools
7. **Data Management**

   - Graph backup/restore
   - Incremental updates
   - Document versioning
   - Bulk import/export

### Low Priority

8. **UI/UX**

   - Web interface for document upload
   - Interactive graph visualization
   - Search interface demo
   - Admin dashboard
9. **Integration**

   - Slack/Teams bots
   - REST API SDKs
   - Webhook support
   - Third-party integrations

---

## Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest -v

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest test_system.py::TestEfficientNLPProcessor -v
pytest test_system.py::TestUnifiedSearchPipeline -v
```

### Test Coverage

- ✅ **NLP Processing** - Entity extraction, coreference resolution
- ✅ **Embedding Operations** - Batch processing, caching
- ✅ **Search Pipeline** - Vector search, reranking, answer generation
- ✅ **Database Operations** - Neo4j connectivity, queries
- ✅ **Error Handling** - Edge cases, invalid inputs
- ✅ **Performance** - Batch vs individual processing
- ✅ **Question Classification** - Accuracy testing for BROAD/CHUNK/GRAPH_ANALYTICAL/OUT_OF_SCOPE classification
- ✅ **Cypher Query Processing** - Multi-strategy query generation with embedding-based vector search for entity matching, text-based fallback, and result formatting
- ✅ **Map-Reduce Processing** - Aggregation accuracy and partial failure handling
- ✅ **Classification Routing** - Integration tests for question classification and routing

---

## Performance Optimizations

### Implemented Optimizations

| Optimization                    | Impact             | Implementation                           |
| ------------------------------- | ------------------ | ---------------------------------------- |
| **Batch Embedding**       | 10x faster         | Batch API calls with caching             |
| **Vector Indexes**        | 100x faster search | Native Neo4j vector indexes              |
| **Entity Vector Search**  | Better accuracy    | Embedding-based semantic matching for entities |
| **LLM-based NER**         | Flexible           | Configurable (Gemini, OpenAI-compatible) |
| **Async Operations**      | Non-blocking       | Full async/await pipeline                |
| **Intelligent Caching**   | Reduced API calls  | TTL-based embedding cache                |
| **Graph-Aware Reranking** | Better relevance   | Entity overlap + community scoring       |
| **MMR Reranking**         | Better diversity   | Maximal Marginal Relevance               |

### Performance Metrics

- **Document Processing**: ~0.5-2 seconds per document (depending on size)
- **Entity Extraction**: ~0.1-0.5 seconds per chunk (LLM-based, model dependent)
- **Search Response Time**: 1-3 seconds average (varies by question type and classification method)
- **Embedding Generation**: Batch of 10 in ~0.5 seconds (cached)
- **Question Classification**: < 0.5 seconds (MCP or heuristic), 1-2 seconds (direct LLM)
- **Cypher Query Processing**: 2-5 seconds (includes term extraction, embedding generation, vector-based entity matching, multi-strategy execution, and answer formatting)

---

## Additional Resources

### Project Structure

```
graph-rag/
├── main.py                 # FastAPI application entry point
├── config.py               # Centralized configuration module
├── resilience.py           # Resilience patterns (circuit breakers, retries)
├── logging_config.py       # Standardized structured logging
├── document_api.py         # Document processing endpoints
├── search_api.py           # Search endpoints
├── unified_search.py       # Core search pipeline logic with classification routing
├── question_classifier.py  # Question classification (MCP, heuristic, LLM)
├── map_reduce.py           # Map-reduce processing for broad questions
├── mcp_classifier_client.py # MCP classifier client
├── mcp_classifier_server.py # MCP classifier server
├── mcp_neo4j_cypher.py     # MCP Neo4j Cypher client for graph analytical queries
├── utils.py                # NLP, embeddings, utilities
├── test_system.py          # Comprehensive test suite
├── requirements.txt        # Python dependencies
├── docker-compose.yaml     # Docker Compose configuration for all services
├── Dockerfile              # Main Graph RAG API Dockerfile
├── Dockerfile.mcp          # MCP classifier server Dockerfile
├── .env                    # Environment configuration (not in git)
├── .env.example            # Environment configuration template
└── README.md              # This file
```

**New Architecture Components:**
- `config.py`: Centralized configuration loading, validation, and documentation
- `resilience.py`: Circuit breakers and retry logic for external service calls
- `logging_config.py`: Standardized structured logging with context fields
- `mcp_neo4j_cypher.py`: MCP Neo4j Cypher client for graph analytical queries and iterative query refinement

All modules now use these infrastructure components for consistency and reliability.

### Key Dependencies

- **FastAPI** - Modern async web framework
- **Neo4j Driver** - Graph database connectivity
- **HTTPX** - Async HTTP client for LLM and embedding APIs
- **PyPDF** - PDF text extraction
- **python-docx** - Word document parsing
- **BlingFire** - Sentence boundary detection
- **NumPy** - Numerical computations
- **MCP (Model Context Protocol)** - For question classification (optional)

### Documentation

- **API Documentation**: Available at `/docs` (Swagger UI)
- **Alternative Docs**: Available at `/redoc` (ReDoc)
- **Health Check**: `/health` for system status
- **API Info**: `/api/info` for capabilities

---

## Support & Contribution

### Getting Help

- **API Documentation**: Visit `/docs` for interactive API documentation
- **Health Check**: Monitor system status at `/health`
- **Analytics**: View usage statistics at `/api/search/search_analytics`
- **Logs**: Check application logs for debugging

### Development

#### Docker Development

For development with Docker, you can mount the code as a volume for live reloading:

```yaml
# In docker-compose.yaml, add volumes to graph-rag service:
volumes:
  - ./logs:/app/logs
  - .:/app  # Mount code for development
  - /app/__pycache__  # Exclude cache
```

Then set environment variables in `.env`:
```bash
RELOAD=true
LOG_LEVEL=DEBUG
```

#### Local Development

For local development without Docker:

```bash
# Set environment variables
export RELOAD=true
export LOG_LEVEL=DEBUG

# Or create a .env file with:
# RELOAD=true
# LOG_LEVEL=DEBUG

# Run the application
python main.py
```

### Docker Commands

Common Docker Compose commands:

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f graph-rag
docker-compose logs -f neo4j
docker-compose logs -f mcp-classifier

# Rebuild and restart a service
docker-compose up -d --build graph-rag

# Check service status
docker-compose ps

# Execute commands in a container
docker-compose exec graph-rag python -c "from utils import nlp_processor; print('OK')"

# Stop and remove containers, networks, and volumes
docker-compose down -v
```

---

## License

See LICENSE file for details.

---

## Version History

- **v2.3.0** - Enhanced entity matching: Embedding-based vector search for fuzzy entity matching in Cypher queries with high similarity thresholds, automatic text-based fallback, and optimized batch embedding generation for entities during document processing.
- **v2.2.0** - Graph analytical queries: MCP Neo4j Cypher integration with multi-strategy query generation, fuzzy entity matching, iterative refinement, and comprehensive explainability. Added GRAPH_ANALYTICAL question type for complex graph queries, aggregations, and multi-hop reasoning.
- **v2.1.0** - Production-ready resilience patterns: centralized configuration (`config.py`), circuit breakers and automatic retries (`resilience.py`), structured logging (`logging_config.py`). All external service calls now use resilience patterns to handle transient failures gracefully without user-facing errors.
- **v2.0.0** - Unified search pipeline with intelligent question classification, map-reduce for broad questions, MCP classifier integration, graph-aware reranking, MMR diversity, cross-document entity merging
- **v1.0.0** - Initial release with basic document processing and search

---

## Resilience & Reliability

### Circuit Breakers

The system uses circuit breakers to prevent cascading failures:
- **Per-service breakers**: Separate breakers for LLM, embedding, NER, coreference, Neo4j, and MCP classifier
- **Three states**: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery)
- **Automatic recovery**: Circuit automatically transitions back to CLOSED after successful requests
- **Configurable thresholds**: Adjust failure/success thresholds and timeout periods via environment variables

### Automatic Retries

All external service calls include automatic retry logic:
- **Exponential backoff**: Delays increase exponentially between retries (e.g., 1s, 2s, 4s)
- **Configurable attempts**: Default 3 retries, configurable per service
- **Smart retryable errors**: Only retries on transient failures, not permanent errors
- **Timeout protection**: Requests timeout after configured duration to prevent hanging

### Structured Logging

Consistent logging across all modules:
- **Context fields**: Request IDs, operation names, service context automatically included
- **Structured format**: Consistent schema for easy parsing and analysis
- **JSON option**: Can output JSON-formatted logs for production log aggregation (set `LOG_JSON=true`)
- **Error context**: All errors logged with full context for easier debugging in production

### Configuration Management

Centralized configuration provides:
- **Single source of truth**: All settings in `config.py` with validation
- **Type safety**: Type-checked configuration with Pydantic
- **Documentation**: Settings documented with defaults and descriptions
- **Runtime validation**: Configuration validated on startup to catch errors early

---

**Built with ❤️ for intelligent document understanding and retrieval.**
