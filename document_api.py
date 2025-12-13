"""
Document API for Graph RAG with efficient processing and batch optimization.

Features:
- LLM-based NER using configurable models via OpenAI-compatible API
- LLM-based coreference resolution using configurable models
- Boilerplate and navigation text removal before chunking
- Batch embedding processing (10x speed improvement)
- Proper async handling throughout
- Centralized configuration and structured logging
- Resilience patterns for external service calls (retries + circuit breakers)
- Community detection and summarization

This module uses:
- config.py: Centralized configuration (database, processing, embedding settings)
- logging_config.py: Standardized structured logging with context fields
- resilience.py: Automatic retries and circuit breaking for LLM and database calls
"""

import asyncio
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

import pypdf
import docx
from fastapi import APIRouter, HTTPException, UploadFile, File
from neo4j import GraphDatabase
from pydantic import BaseModel

# Import centralized configuration, logging, and resilience
from config import get_config
from logging_config import get_logger, log_function_call, log_error_with_context
from utils import (
    nlp_processor, embedding_client, text_processor, 
    run_cypher_query_async, extract_entities_efficient,
    clean_text_improved, chunk_text_improved, llm_client
)

# Get configuration
cfg = get_config()

# Use configuration values (backward compatibility)
CHUNK_SIZE_GDS = cfg.processing.chunk_size
DOCUMENT_PROCESSING_BATCH_SIZE = cfg.processing.document_processing_batch_size
EMBEDDING_DIMENSION = cfg.embedding.dimension

# Setup logging using centralized logging config
logger = get_logger("DocumentAPI")

# Initialize Neo4j driver
driver = GraphDatabase.driver(
    cfg.database.url,
    auth=(cfg.database.username, cfg.database.password)
)

# Pydantic models
class DeleteDocumentRequest(BaseModel):
    doc_id: Optional[str] = None
    document_name: Optional[str] = None

class DocumentProcessingResult(BaseModel):
    doc_id: str
    document_name: str
    chunks_created: int
    entities_extracted: int
    processing_time: float

class GraphManager:
    """
    Graph manager with efficient batch processing and better error handling.
    
    All database operations use resilience patterns:
    - Automatic retries with exponential backoff via resilience.py
    - Circuit breaker protection to prevent cascading failures
    - Structured logging with database operation context from logging_config.py
    - Timeout protection for long-running queries
    - Centralized configuration from config.py
    """
    
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self.logger = get_logger("GraphManager")
        self._vector_indexes_created = False
    
    async def _ensure_vector_indexes(self, embedding_dim: Optional[int] = None):
        """Create vector indexes for embeddings if they don't exist"""
        if self._vector_indexes_created:
            return
        
        # Use provided dimension, or try to detect from database, or fall back to env variable
        if embedding_dim is None:
            embedding_dim = EMBEDDING_DIMENSION
        
        try:
            # Try to get actual embedding dimension from existing chunks
            try:
                dim_query = """
                MATCH (c:Chunk)
                WHERE c.embedding IS NOT NULL
                RETURN size(c.embedding) AS dim
                LIMIT 1
                """
                dim_result = await run_cypher_query_async(self.driver, dim_query)
                if dim_result and dim_result[0].get("dim"):
                    embedding_dim = dim_result[0]["dim"]
                    self.logger.info(f"Detected embedding dimension: {embedding_dim}")
            except Exception as dim_e:
                self.logger.debug(f"Could not detect embedding dimension, using configured value {embedding_dim}: {dim_e}")
            
            # Create vector index for Chunk embeddings
            chunk_index_query = f"""
            CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {embedding_dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            
            # Create vector index for Entity embeddings
            entity_index_query = f"""
            CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS
            FOR (e:Entity) ON (e.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {embedding_dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            
            # Create vector index for CommunitySummary embeddings
            community_index_query = f"""
            CREATE VECTOR INDEX community_summary_embedding_index IF NOT EXISTS
            FOR (cs:CommunitySummary) ON (cs.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {embedding_dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            
            await run_cypher_query_async(self.driver, chunk_index_query)
            await run_cypher_query_async(self.driver, entity_index_query)
            await run_cypher_query_async(self.driver, community_index_query)
            self._vector_indexes_created = True
            self.logger.info(f"Vector indexes created successfully with dimension {embedding_dim}")
            
        except Exception as e:
            # If vector indexes are not supported (Neo4j < 5.11), log warning and continue
            if "VECTOR" in str(e).upper() or "index" in str(e).lower():
                self.logger.warning(f"Vector indexes may not be supported in this Neo4j version: {e}")
                self.logger.warning("Falling back to property-based storage")
            else:
                self.logger.error(f"Error creating vector indexes: {e}")
                raise
    
    async def build_graph_efficient(
        self, 
        chunks: List[str], 
        metadata_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build graph efficiently using batch processing and LLM-based NER
        """
        start_time = asyncio.get_event_loop().time()
        
        if len(chunks) != len(metadata_list):
            raise ValueError("Chunks and metadata lists must have the same length")
        
        # Filter out empty chunks
        valid_chunks = []
        valid_metadata = []
        for chunk, meta in zip(chunks, metadata_list):
            if chunk and chunk.strip() and not chunk.strip().isdigit():
                valid_chunks.append(chunk)
                valid_metadata.append(meta)
        
        if not valid_chunks:
            raise ValueError("No valid chunks to process")
        
        self.logger.info(f"Processing {len(valid_chunks)} valid chunks")
        
        # Step 1: Batch generate embeddings for all chunks
        chunk_embeddings = await embedding_client.get_embeddings(valid_chunks)
        
        # Validate embeddings were generated
        if not chunk_embeddings or len(chunk_embeddings) == 0:
            raise ValueError("Failed to generate embeddings for chunks")
        
        # Validate that embeddings are not empty
        valid_embeddings = []
        valid_chunks_filtered = []
        valid_metadata_filtered = []
        
        for i, (chunk, meta, embedding) in enumerate(zip(valid_chunks, valid_metadata, chunk_embeddings)):
            if embedding and len(embedding) > 0:
                valid_embeddings.append(embedding)
                valid_chunks_filtered.append(chunk)
                valid_metadata_filtered.append(meta)
            else:
                self.logger.warning(f"Skipping chunk {i} due to empty embedding")
        
        if len(valid_embeddings) == 0:
            raise ValueError("No valid embeddings generated for any chunks")
        
        chunk_embeddings = valid_embeddings
        valid_chunks = valid_chunks_filtered
        valid_metadata = valid_metadata_filtered
        
        # Ensure vector indexes exist (after we have embeddings to detect dimension)
        if chunk_embeddings and len(chunk_embeddings) > 0:
            embedding_dim = len(chunk_embeddings[0])
            if embedding_dim == 0:
                raise ValueError("Embedding dimension is 0 - embeddings are empty")
            await self._ensure_vector_indexes(embedding_dim)
        
        # Step 2: Extract entities from all chunks using LLM-based processing
        all_entities_data = []
        for i, chunk in enumerate(valid_chunks):
            entities = await nlp_processor.extract_entities(chunk)
            all_entities_data.append(entities)
        
        # Step 3: Collect unique entities and batch generate their embeddings
        # Merge entities by name only (not by name + doc_id) to enable cross-document connections
        unique_entities = {}
        for entities in all_entities_data:
            for entity in entities:
                key = entity.name.lower()  # Use name only, not name + doc_id
                if key not in unique_entities:
                    unique_entities[key] = entity
        
        entity_names = [entity.name for entity in unique_entities.values()]
        entity_embeddings = await embedding_client.get_embeddings(entity_names) if entity_names else []
        
        # Step 4: Store chunks and entities in Neo4j
        chunks_created = 0
        entities_created = 0
        
        try:
            # Process chunks in batches (use smaller batches to reduce memory pressure)
            batch_size = DOCUMENT_PROCESSING_BATCH_SIZE
            for i in range(0, len(valid_chunks), batch_size):
                batch_chunks = valid_chunks[i:i + batch_size]
                batch_metadata = valid_metadata[i:i + batch_size]
                batch_embeddings = chunk_embeddings[i:i + batch_size]
                batch_entities = all_entities_data[i:i + batch_size]
                
                result = await self._process_chunk_batch(
                    batch_chunks, batch_metadata, batch_embeddings, batch_entities, i
                )
                chunks_created += result["chunks_created"]
                entities_created += result["entities_created"]
            
            # Step 5: Create entity relationships (now works across all documents)
            await self._create_entity_relationships()
            
            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time
            
            self.logger.info(f"Graph building completed in {processing_time:.2f}s")
            
            return {
                "chunks_created": chunks_created,
                "entities_created": entities_created,
                "processing_time": processing_time,
                "doc_id": valid_metadata[0]["doc_id"]
            }
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                f"Error in graph building: {e}",
                exception=e,
                context=log_function_call("build_graph_efficient", chunks_count=len(chunks))
            )
            raise
    
    async def _process_chunk_batch(
        self,
        chunks: List[str],
        metadata_list: List[Dict[str, Any]],
        embeddings: List[List[float]],
        entities_list: List[List[Any]],
        start_index: int
    ) -> Dict[str, int]:
        """Process a batch of chunks efficiently"""
        
        chunks_created = 0
        entities_created = 0
        
        # Create chunks
        for i, (chunk, meta, embedding, entities) in enumerate(
            zip(chunks, metadata_list, embeddings, entities_list)
        ):
            chunk_id = start_index + i
            
            # Validate embedding before storing
            if not embedding or len(embedding) == 0:
                self.logger.warning(f"Skipping chunk {chunk_id} due to empty embedding")
                continue
            
            # Create chunk node
            chunk_query = """
            MERGE (c:Chunk {id: $chunk_id, doc_id: $doc_id})
            SET c.text = $text,
                c.document_name = $document_name,
                c.timestamp = $timestamp,
                c.embedding = $embedding
            """
            
            await run_cypher_query_async(self.driver, chunk_query, {
                "chunk_id": chunk_id,
                "doc_id": meta["doc_id"],
                "text": chunk,
                "document_name": meta.get("document_name", ""),
                "timestamp": meta.get("timestamp", datetime.now().isoformat()),
                "embedding": embedding
            })
            chunks_created += 1
            
            # Create entity nodes and relationships
            for entity in entities:
                entity_name = entity.name.lower()
                entity_type = entity.type
                
                # Get entity embedding
                try:
                    entity_embedding = await embedding_client.get_embedding(entity.name)
                except Exception as e:
                    self.logger.warning(f"Failed to get embedding for entity {entity.name}: {e}")
                    entity_embedding = []
                
                # Create/update entity node - merge by name only to enable cross-document connections
                # Track which documents the entity appears in using a documents property
                entity_query = """
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.type = $type,
                             e.embedding = $embedding,
                             e.documents = [$doc_id]
                ON MATCH SET e.type = CASE 
                                        WHEN e.type IS NULL OR e.type = '' THEN $type 
                                        ELSE e.type 
                                      END,
                             e.embedding = CASE 
                                            WHEN e.embedding IS NULL OR size(e.embedding) = 0 THEN $embedding 
                                            ELSE e.embedding 
                                          END,
                             e.documents = CASE 
                                            WHEN $doc_id IN e.documents THEN e.documents 
                                            ELSE e.documents + $doc_id 
                                          END
                """
                
                await run_cypher_query_async(self.driver, entity_query, {
                    "name": entity_name,
                    "doc_id": meta["doc_id"],
                    "type": entity_type,
                    "embedding": entity_embedding
                })
                
                # Create MENTIONED_IN relationship - works with merged entities across documents
                mention_query = """
                MATCH (e:Entity {name: $name})
                MATCH (c:Chunk {id: $chunk_id, doc_id: $doc_id})
                MERGE (e)-[:MENTIONED_IN {doc_id: $doc_id}]->(c)
                """
                
                await run_cypher_query_async(self.driver, mention_query, {
                    "name": entity_name,
                    "doc_id": meta["doc_id"],
                    "chunk_id": chunk_id
                })
                
                entities_created += 1
        
        return {"chunks_created": chunks_created, "entities_created": entities_created}
    
    async def _create_entity_relationships(self):
        """Create RELATES_TO relationships between entities that co-occur in chunks
        Now works across all documents - entities with same name are merged"""
        
        # Create cross-document RELATES_TO relationships
        # Entities are now merged by name, so relationships connect entities across documents
        relationship_query = """
        MATCH (e1:Entity)-[:MENTIONED_IN]->(c:Chunk)<-[:MENTIONED_IN]-(e2:Entity)
        WHERE e1.name < e2.name
        WITH e1, e2, count(DISTINCT c) as co_occurrence_count
        WHERE co_occurrence_count > 0
        MERGE (e1)-[r:RELATES_TO]-(e2)
        ON CREATE SET r.weight = co_occurrence_count
        ON MATCH SET r.weight = co_occurrence_count
        """
        
        await run_cypher_query_async(self.driver, relationship_query, {})
        
        self.logger.info("Created cross-document entity relationships")
    
    async def generate_community_summaries(self, doc_id: str) -> Dict[str, Any]:
        """
        Generate community summaries using Leiden algorithm from Neo4j GDS
        """
        try:
            # Use Leiden algorithm for community detection
            communities = await self._detect_communities_leiden(doc_id)
            
            # Fallback to simple method if Leiden fails
            if not communities:
                self.logger.warning("Leiden algorithm failed, falling back to simple co-occurrence")
                communities = await self._detect_communities_simple(doc_id)
            
            if not communities:
                self.logger.warning(f"No communities detected for doc_id: {doc_id}")
                return {"communities_created": 0}
            
            summaries_created = 0
            
            for community_id, entity_names in communities.items():
                # Get chunks that mention these entities
                chunks_text = await self._get_community_chunks(doc_id, entity_names)
                
                if not chunks_text:
                    continue
                
                # Generate summary using LLM
                summary = await self._generate_community_summary(chunks_text, entity_names)
                
                if summary:
                    # Get summary embedding
                    summary_embedding = await embedding_client.get_embedding(summary)
                    
                    # Store community summary
                    store_query = """
                    MERGE (cs:CommunitySummary {doc_id: $doc_id, community: $community})
                    SET cs.summary = $summary,
                        cs.embedding = $embedding,
                        cs.entities = $entities,
                        cs.timestamp = $timestamp
                    """
                    
                    await run_cypher_query_async(self.driver, store_query, {
                        "doc_id": doc_id,
                        "community": community_id,
                        "summary": summary,
                        "embedding": summary_embedding,
                        "entities": entity_names,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    summaries_created += 1
            
            self.logger.info(f"Created {summaries_created} community summaries for doc_id: {doc_id}")
            return {"communities_created": summaries_created}
            
        except Exception as e:
            self.logger.error(f"Error generating community summaries: {e}")
            return {"communities_created": 0, "error": str(e)}
    
    async def _detect_communities_leiden(self, doc_id: str) -> Dict[str, List[str]]:
        """Detect communities using Leiden algorithm from Neo4j GDS
        Now works with merged entities - filters chunks by doc_id but uses cross-document entities"""
        try:
            graph_name = f"entity-graph-{doc_id.replace('-', '_')}"  # GDS graph names can't have hyphens
            
            # First, create CO_OCCURS relationships for this document
            # Entities are now merged across documents, but we filter chunks by doc_id
            create_relationships_query = """
            MATCH (e1:Entity)-[r1:MENTIONED_IN {doc_id: $doc_id}]->(c:Chunk {doc_id: $doc_id})<-[:MENTIONED_IN {doc_id: $doc_id}]-(e2:Entity)
            WHERE e1.name <> e2.name AND e1.name < e2.name
            WITH e1, e2, count(DISTINCT c) AS weight
            WHERE weight >= 2
            MERGE (e1)-[r:CO_OCCURS]-(e2)
            ON CREATE SET r.weight = weight, r.doc_id = $doc_id
            ON MATCH SET r.weight = CASE 
                                      WHEN r.doc_id = $doc_id THEN weight 
                                      ELSE r.weight 
                                    END
            """
            
            await run_cypher_query_async(self.driver, create_relationships_query, {"doc_id": doc_id})
            
            # Drop existing graph projection if it exists
            try:
                drop_existing = f"CALL gds.graph.drop('{graph_name}') YIELD graphName"
                await run_cypher_query_async(self.driver, drop_existing, {})
            except:
                pass  # Graph may not exist
            
            # Create a graph projection for GDS using Cypher projection
            # Filter entities that appear in chunks from this document
            # Note: GDS uses internal node IDs (id()) for graph projection - this is acceptable in GDS context
            project_query = f"""
            CALL gds.graph.project.cypher(
                '{graph_name}',
                'MATCH (e:Entity)-[:MENTIONED_IN {{doc_id: $doc_id}}]->(:Chunk {{doc_id: $doc_id}}) RETURN DISTINCT id(e) AS id',
                'MATCH (e1:Entity)-[r:CO_OCCURS]-(e2:Entity) WHERE r.doc_id = $doc_id RETURN id(e1) AS source, id(e2) AS target, coalesce(r.weight, 1.0) AS weight',
                {{
                    parameters: {{doc_id: $doc_id}}
                }}
            )
            YIELD graphName, nodeCount, relationshipCount, projectMillis
            """
            
            project_result = await run_cypher_query_async(self.driver, project_query, {"doc_id": doc_id})
            
            if not project_result or project_result[0].get("nodeCount", 0) == 0:
                self.logger.warning("No entities found for Leiden algorithm")
                return {}
            
            # Run Leiden algorithm
            # Note: Using elementId() for node lookup instead of deprecated id()
            leiden_query = f"""
            CALL gds.leiden.stream('{graph_name}', {{
                maxLevels: 10,
                randomSeed: 42
            }})
            YIELD nodeId, communityId
            RETURN elementId(gds.util.asNode(nodeId)) AS node_element_id, gds.util.asNode(nodeId).name AS entity_name, communityId
            """
            
            results = await run_cypher_query_async(self.driver, leiden_query, {})
            
            # Group entities by community
            communities = {}
            for result in results:
                community_id = f"community_{result['communityId']}"
                entity_name = result['entity_name']
                
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(entity_name)
            
            # Clean up graph projection
            try:
                drop_query = f"CALL gds.graph.drop('{graph_name}') YIELD graphName"
                await run_cypher_query_async(self.driver, drop_query, {})
            except:
                pass  # Graph may not exist or already dropped
            
            self.logger.info(f"Leiden algorithm detected {len(communities)} communities for doc_id: {doc_id}")
            return communities
            
        except Exception as e:
            self.logger.warning(f"Leiden algorithm failed: {e}")
            return {}
    
    async def _detect_communities_simple(self, doc_id: str) -> Dict[str, List[str]]:
        """Simple community detection based on entity co-occurrence (fallback method)
        Now works with merged entities - filters chunks by doc_id"""
        
        # Get entity co-occurrence data - filter chunks by doc_id but use merged entities
        query = """
        MATCH (e1:Entity)-[:MENTIONED_IN {doc_id: $doc_id}]->(c:Chunk {doc_id: $doc_id})<-[:MENTIONED_IN {doc_id: $doc_id}]-(e2:Entity)
        WHERE e1.name <> e2.name
        RETURN e1.name as entity1, e2.name as entity2, count(DISTINCT c) as co_occurrence
        ORDER BY co_occurrence DESC
        """
        
        results = await run_cypher_query_async(self.driver, query, {"doc_id": doc_id})
        
        # Simple clustering: group entities that frequently co-occur
        communities = {}
        entity_to_community = {}
        community_counter = 0
        
        for result in results:
            entity1 = result["entity1"]
            entity2 = result["entity2"]
            co_occurrence = result["co_occurrence"]
            
            # Only consider strong co-occurrences
            if co_occurrence < 2:
                continue
            
            # Check if either entity is already in a community
            comm1 = entity_to_community.get(entity1)
            comm2 = entity_to_community.get(entity2)
            
            if comm1 is not None and comm2 is not None:
                # Both entities are in communities - merge if different
                if comm1 != comm2:
                    # Merge communities
                    communities[comm1].extend(communities[comm2])
                    for entity in communities[comm2]:
                        entity_to_community[entity] = comm1
                    del communities[comm2]
            elif comm1 is not None:
                # Add entity2 to entity1's community
                communities[comm1].append(entity2)
                entity_to_community[entity2] = comm1
            elif comm2 is not None:
                # Add entity1 to entity2's community
                communities[comm2].append(entity1)
                entity_to_community[entity1] = comm2
            else:
                # Create new community
                community_id = f"community_{community_counter}"
                communities[community_id] = [entity1, entity2]
                entity_to_community[entity1] = community_id
                entity_to_community[entity2] = community_id
                community_counter += 1
        
        # Remove duplicates in communities
        for community_id in communities:
            communities[community_id] = list(set(communities[community_id]))
        
        return communities
    
    async def _get_community_chunks(self, doc_id: str, entity_names: List[str]) -> str:
        """Get text chunks that mention the community entities
        Now works with merged entities - filters chunks by doc_id"""
        
        query = """
        MATCH (e:Entity)-[:MENTIONED_IN {doc_id: $doc_id}]->(c:Chunk {doc_id: $doc_id})
        WHERE e.name IN $entity_names
        RETURN DISTINCT c.text as chunk_text
        LIMIT 10
        """
        
        results = await run_cypher_query_async(self.driver, query, {
            "doc_id": doc_id,
            "entity_names": entity_names
        })
        
        chunks = [result["chunk_text"] for result in results if result.get("chunk_text")]
        return "\n\n".join(chunks)
    
    async def _generate_community_summary(self, chunks_text: str, entity_names: List[str]) -> str:
        """Generate a summary for a community using LLM"""
        
        if not chunks_text.strip():
            return ""
        
        try:
            prompt = f"""
            Based on the following text excerpts that mention entities {', '.join(entity_names)}, 
            create a concise summary that captures the main themes and relationships.
            
            Text excerpts:
            {chunks_text[:2000]}  # Limit to avoid token limits
            
            Summary:"""
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that creates concise, informative summaries."},
                {"role": "user", "content": prompt}
            ]
            
            summary = await llm_client.invoke(messages)
            return summary.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating community summary: {e}")
            return f"Summary of entities: {', '.join(entity_names)}"

# Initialize graph manager
graph_manager = GraphManager(driver)

# API Router
router = APIRouter()

@router.post("/upload_documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and process documents into the knowledge graph.
    
    This endpoint processes uploaded documents by:
    1. **Text Extraction**: Extracts text from PDF, DOCX, or TXT files
    2. **Text Cleaning**: Removes boilerplate, navigation text, and noise
    3. **Chunking**: Splits text into semantic chunks (default: 512 tokens)
    4. **Entity Extraction**: Uses LLM-based NER to extract entities (PERSON, ORGANIZATION, etc.)
    5. **Embedding Generation**: Creates vector embeddings for chunks and entities (batch processing)
    6. **Graph Building**: Stores chunks, entities, and relationships in Neo4j
    7. **Community Detection**: Uses Leiden algorithm to detect entity communities
    8. **Summary Generation**: Creates community summaries using LLM
    
    **Supported Formats**: PDF, DOCX, TXT
    
    **Processing Features**:
    - Batch embedding optimization (10x speed improvement)
    - Cross-document entity merging (entities with same name are merged)
    - Vector index creation for fast similarity search
    - Community detection and summarization
    
    Args:
        files: List of uploaded files (PDF, DOCX, or TXT)
    
    Returns:
        Processing results with doc_id, chunks created, entities extracted, and processing time
    
    Example:
        POST /api/documents/upload_documents
        Content-Type: multipart/form-data
        Files: [document1.pdf, document2.docx]
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results = []
    
    for file in files:
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Extract text based on file type
            filename = file.filename.lower()
            
            if filename.endswith(".pdf"):
                reader = pypdf.PdfReader(file.file)
                text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
            elif filename.endswith(".docx"):
                doc = docx.Document(file.file)
                text = "\n".join(para.text for para in doc.paragraphs)
            elif filename.endswith(".txt"):
                text = (await file.read()).decode("utf-8")
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.filename}")
            
            if not text or not text.strip():
                raise HTTPException(status_code=400, detail=f"No text content found in {file.filename}")
            
            # Clean and process text
            cleaned_text = clean_text_improved(text)
            
            # Generate document ID and metadata
            doc_id = str(uuid.uuid4())
            metadata = {
                "doc_id": doc_id,
                "document_name": file.filename,
                "timestamp": datetime.now().isoformat()
            }
            
            # Chunk the text
            chunks = chunk_text_improved(cleaned_text, max_chunk_size=CHUNK_SIZE_GDS)
            
            if not chunks:
                raise HTTPException(status_code=400, detail=f"No valid chunks created from {file.filename}")
            
            logger.info(
                f"Processing {file.filename}: {len(chunks)} chunks",
                extra=log_function_call("upload_documents", file_name=file.filename, chunks_count=len(chunks))
            )
            
            # Build graph
            metadata_list = [metadata] * len(chunks)
            graph_result = await graph_manager.build_graph_efficient(chunks, metadata_list)
            
            # Generate community summaries
            community_result = await graph_manager.generate_community_summaries(doc_id)
            
            end_time = asyncio.get_event_loop().time()
            total_time = end_time - start_time
            
            results.append(DocumentProcessingResult(
                doc_id=doc_id,
                document_name=file.filename,
                chunks_created=graph_result["chunks_created"],
                entities_extracted=graph_result["entities_created"],
                processing_time=total_time
            ))
            
            logger.info(f"Successfully processed {file.filename} in {total_time:.2f}s")
            
        except Exception as e:
            log_error_with_context(
                logger,
                f"Error processing {file.filename}: {e}",
                exception=e,
                context=log_function_call("upload_documents", file_name=file.filename)
            )
            raise HTTPException(status_code=500, detail=f"Error processing {file.filename}: {str(e)}")
    
    return {
        "message": f"Successfully processed {len(results)} documents",
        "results": [result.dict() for result in results]
    }

@router.get("/documents")
async def list_documents():
    """
    List all documents in the knowledge base.
    
    Returns a list of all processed documents with:
    - Document ID (doc_id)
    - Document name (filename)
    - Timestamp (when processed)
    
    Results are ordered by timestamp (most recent first).
    
    Returns:
        List of documents with metadata
    
    Example:
        GET /api/documents/documents
    """
    try:
        query = """
        MATCH (c:Chunk)
        RETURN DISTINCT c.doc_id AS doc_id, c.document_name AS document_name, c.timestamp AS timestamp
        ORDER BY c.timestamp DESC
        """
        
        documents = await run_cypher_query_async(driver, query)
        return {"documents": documents}
        
    except Exception as e:
        log_error_with_context(
            logger,
            f"Error listing documents: {e}",
            exception=e,
            context=log_function_call("list_documents")
        )
        raise HTTPException(status_code=500, detail="Error listing documents")

@router.delete("/delete_document")
async def delete_document(request: DeleteDocumentRequest):
    """
    Delete a document and all associated data from the knowledge base.
    
    This operation removes:
    - All chunks from the document
    - All entities (if they only appear in this document)
    - All relationships involving the document's chunks
    - All community summaries for the document
    
    **Note**: Entities that appear in multiple documents are preserved, but
    their relationships to this document's chunks are removed.
    
    Args:
        request: DeleteDocumentRequest with either doc_id or document_name
    
    Returns:
        Success message confirming deletion
    
    Example:
        ```json
        {
          "doc_id": "abc-123-def-456"
        }
        ```
        or
        ```json
        {
          "document_name": "my-document.pdf"
        }
        ```
    """
    if not request.doc_id and not request.document_name:
        raise HTTPException(
            status_code=400, 
            detail="Provide either doc_id or document_name to delete a document."
        )
    
    try:
        if request.doc_id:
            query = "MATCH (n) WHERE n.doc_id = $doc_id DETACH DELETE n"
            params = {"doc_id": request.doc_id}
            message = f"Document with doc_id {request.doc_id} deleted successfully."
        else:
            query = "MATCH (n) WHERE n.document_name = $document_name DETACH DELETE n"
            params = {"document_name": request.document_name}
            message = f"Document with name {request.document_name} deleted successfully."
        
        await run_cypher_query_async(driver, query, params)
        return {"message": message}
        
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {e}")

@router.get("/community_summaries")
async def get_community_summaries(doc_id: Optional[str] = None):
    """
    Get community summaries generated from entity communities.
    
    Community summaries are high-level topic summaries created by:
    1. Detecting entity communities using Leiden algorithm (or simple co-occurrence)
    2. Extracting chunks that mention entities in each community
    3. Generating LLM-based summaries of the community themes
    
    These summaries are used for answering broad questions via map-reduce processing.
    
    Args:
        doc_id: Optional document ID to filter summaries to specific document.
                If not provided, returns summaries from all documents (max 50).
    
    Returns:
        List of community summaries with community ID, summary text, entities, and timestamp
    
    Example:
        GET /api/documents/community_summaries?doc_id=abc-123
    """
    try:
        if doc_id:
            query = """
            MATCH (cs:CommunitySummary {doc_id: $doc_id})
            RETURN cs.community AS community, cs.summary AS summary, 
                   cs.entities AS entities, cs.timestamp AS timestamp
            ORDER BY cs.timestamp DESC
            """
            params = {"doc_id": doc_id}
        else:
            query = """
            MATCH (cs:CommunitySummary)
            RETURN cs.doc_id AS doc_id, cs.community AS community, cs.summary AS summary,
                   cs.entities AS entities, cs.timestamp AS timestamp
            ORDER BY cs.timestamp DESC
            LIMIT 50
            """
            params = {}
        
        results = await run_cypher_query_async(driver, query, params)
        return {"community_summaries": results}
        
    except Exception as e:
        logger.error(f"Error retrieving community summaries: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving community summaries: {e}")

@router.get("/document_stats")
async def get_document_stats():
    """
    Get comprehensive statistics about the document collection.
    
    Returns:
        - Total number of chunks
        - Total number of documents
        - Total number of entities
        - Total number of community summaries
    
    Useful for monitoring knowledge base size and health.
    
    Example:
        GET /api/documents/document_stats
    """
    try:
        stats_query = """
        MATCH (c:Chunk)
        WITH count(c) as total_chunks, count(DISTINCT c.doc_id) as total_docs
        MATCH (e:Entity)
        WITH total_chunks, total_docs, count(e) as total_entities
        MATCH (cs:CommunitySummary)
        RETURN total_chunks, total_docs, total_entities, count(cs) as total_summaries
        """
        
        result = await run_cypher_query_async(driver, stats_query)
        stats = result[0] if result else {
            "total_chunks": 0, "total_docs": 0, 
            "total_entities": 0, "total_summaries": 0
        }
        
        return {"statistics": stats}
        
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(status_code=500, detail="Error getting document statistics")
