"""
Improved Document API with efficient NLP processing and better async handling.

This module addresses the key issues:
- Replaces LLM-based NER with efficient spaCy processing
- Implements batch embedding processing
- Fixes async/blocking issues
- Better error handling and logging
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

import PyPDF2
import docx
from fastapi import APIRouter, HTTPException, UploadFile, File
from neo4j import GraphDatabase
from pydantic import BaseModel
from dotenv import load_dotenv

from improved_utils import (
    nlp_processor, embedding_client, text_processor, 
    run_cypher_query_async, extract_entities_efficient,
    clean_text_improved, chunk_text_improved
)

load_dotenv()

# Configuration
DB_URL = os.getenv("DB_URL")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
CHUNK_SIZE_GDS = int(os.getenv("CHUNK_SIZE_GDS", "512"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Setup logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("ImprovedDocumentAPI")

# Initialize Neo4j driver
driver = GraphDatabase.driver(DB_URL, auth=(DB_USERNAME, DB_PASSWORD))

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

class ImprovedGraphManager:
    """
    Improved graph manager with efficient batch processing and better error handling
    """
    
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self.logger = logging.getLogger("ImprovedGraphManager")
    
    async def build_graph_efficient(
        self, 
        chunks: List[str], 
        metadata_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build graph efficiently using batch processing and spaCy NER
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
        
        # Step 2: Extract entities from all chunks using efficient spaCy processing
        all_entities_data = []
        for i, chunk in enumerate(valid_chunks):
            entities = nlp_processor.extract_entities(chunk)
            all_entities_data.append(entities)
        
        # Step 3: Collect unique entities and batch generate their embeddings
        unique_entities = {}
        for entities in all_entities_data:
            for entity in entities:
                key = (entity.name.lower(), valid_metadata[0]["doc_id"])  # Include doc_id in key
                if key not in unique_entities:
                    unique_entities[key] = entity
        
        entity_names = [entity.name for entity in unique_entities.values()]
        entity_embeddings = await embedding_client.get_embeddings(entity_names) if entity_names else []
        
        # Step 4: Store chunks and entities in Neo4j
        chunks_created = 0
        entities_created = 0
        
        try:
            # Process chunks in batches
            batch_size = 10
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
            
            # Step 5: Create entity relationships
            await self._create_entity_relationships(valid_metadata[0]["doc_id"])
            
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
            self.logger.error(f"Error in graph building: {e}")
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
                
                # Create/update entity node
                entity_query = """
                MERGE (e:Entity {name: $name, doc_id: $doc_id})
                SET e.type = $type,
                    e.embedding = $embedding
                """
                
                await run_cypher_query_async(self.driver, entity_query, {
                    "name": entity_name,
                    "doc_id": meta["doc_id"],
                    "type": entity_type,
                    "embedding": entity_embedding
                })
                
                # Create MENTIONED_IN relationship
                mention_query = """
                MATCH (e:Entity {name: $name, doc_id: $doc_id})
                MATCH (c:Chunk {id: $chunk_id, doc_id: $doc_id})
                MERGE (e)-[:MENTIONED_IN]->(c)
                """
                
                await run_cypher_query_async(self.driver, mention_query, {
                    "name": entity_name,
                    "doc_id": meta["doc_id"],
                    "chunk_id": chunk_id
                })
                
                entities_created += 1
        
        return {"chunks_created": chunks_created, "entities_created": entities_created}
    
    async def _create_entity_relationships(self, doc_id: str):
        """Create RELATES_TO relationships between entities that co-occur in chunks"""
        
        relationship_query = """
        MATCH (e1:Entity {doc_id: $doc_id})-[:MENTIONED_IN]->(c:Chunk)<-[:MENTIONED_IN]-(e2:Entity {doc_id: $doc_id})
        WHERE e1.name < e2.name  // Avoid duplicate relationships
        WITH e1, e2, count(c) as co_occurrence_count
        WHERE co_occurrence_count > 0
        MERGE (e1)-[r:RELATES_TO]-(e2)
        SET r.weight = co_occurrence_count
        """
        
        await run_cypher_query_async(self.driver, relationship_query, {"doc_id": doc_id})
        self.logger.info(f"Created entity relationships for doc_id: {doc_id}")
    
    async def generate_community_summaries(self, doc_id: str) -> Dict[str, Any]:
        """
        Generate community summaries using improved clustering and summarization
        """
        try:
            # Use simple clustering based on entity co-occurrence
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
    
    async def _detect_communities_simple(self, doc_id: str) -> Dict[str, List[str]]:
        """Simple community detection based on entity co-occurrence"""
        
        # Get entity co-occurrence data
        query = """
        MATCH (e1:Entity {doc_id: $doc_id})-[:MENTIONED_IN]->(c:Chunk)<-[:MENTIONED_IN]-(e2:Entity {doc_id: $doc_id})
        WHERE e1.name <> e2.name
        RETURN e1.name as entity1, e2.name as entity2, count(c) as co_occurrence
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
        """Get text chunks that mention the community entities"""
        
        query = """
        MATCH (e:Entity {doc_id: $doc_id})-[:MENTIONED_IN]->(c:Chunk)
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
            from improved_utils import llm_client
            
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
graph_manager = ImprovedGraphManager(driver)

# API Router
router = APIRouter()

@router.post("/upload_documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and process documents with improved efficiency and error handling
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
                reader = PyPDF2.PdfReader(file.file)
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
            
            # Optional: Apply coreference resolution for better entity extraction
            # resolved_text = nlp_processor.resolve_coreferences(cleaned_text)
            resolved_text = cleaned_text  # Skip for now to improve performance
            
            # Generate document ID and metadata
            doc_id = str(uuid.uuid4())
            metadata = {
                "doc_id": doc_id,
                "document_name": file.filename,
                "timestamp": datetime.now().isoformat()
            }
            
            # Chunk the text
            chunks = chunk_text_improved(resolved_text, max_chunk_size=CHUNK_SIZE_GDS)
            
            if not chunks:
                raise HTTPException(status_code=400, detail=f"No valid chunks created from {file.filename}")
            
            logger.info(f"Processing {file.filename}: {len(chunks)} chunks")
            
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
            logger.error(f"Error processing {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing {file.filename}: {str(e)}")
    
    return {
        "message": f"Successfully processed {len(results)} documents",
        "results": [result.dict() for result in results]
    }

@router.get("/documents")
async def list_documents():
    """List all documents with proper async handling"""
    try:
        query = """
        MATCH (c:Chunk)
        RETURN DISTINCT c.doc_id AS doc_id, c.document_name AS document_name, c.timestamp AS timestamp
        ORDER BY c.timestamp DESC
        """
        
        documents = await run_cypher_query_async(driver, query)
        return {"documents": documents}
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Error listing documents")

@router.delete("/delete_document")
async def delete_document(request: DeleteDocumentRequest):
    """Delete document with proper async handling"""
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
    """Get community summaries with proper filtering"""
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
    """Get statistics about the document collection"""
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
