"""
Unified Search Pipeline for Graph RAG

This module implements a simplified, efficient search pipeline that replaces the complex
multi-endpoint approach with a single, flexible search function. It addresses the key issues:
- Relevant chunk retrieval based on vector similarity
- Proper context filtering and ranking
- Efficient batch processing
- Better error handling and logging
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv

from improved_utils import (
    nlp_processor, llm_client, embedding_client, text_processor,
    cosine_similarity, run_cypher_query_async
)

load_dotenv()

# Configuration
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))
MAX_CHUNKS_PER_ANSWER = int(os.getenv("MAX_CHUNKS_PER_ANSWER", "7"))
MAX_COMMUNITY_SUMMARIES = int(os.getenv("MAX_COMMUNITY_SUMMARIES", "3"))
SIMILARITY_THRESHOLD_CHUNKS = float(os.getenv("SIMILARITY_THRESHOLD_CHUNKS", "0.4"))
SIMILARITY_THRESHOLD_ENTITIES = float(os.getenv("SIMILARITY_THRESHOLD_ENTITIES", "0.6"))

logger = logging.getLogger("UnifiedSearch")

class SearchScope(Enum):
    """Search scope options"""
    GLOBAL = "global"  # Search across all documents
    LOCAL = "local"    # Search within specific document(s)
    HYBRID = "hybrid"  # Combine global and local approaches

@dataclass
class SearchResult:
    """Structured search result"""
    answer: str
    relevant_chunks: List[Dict[str, Any]]
    community_summaries: List[Dict[str, Any]]
    entities_found: List[Dict[str, Any]]
    confidence_score: float
    search_metadata: Dict[str, Any]

@dataclass
class RetrievedChunk:
    """Chunk with relevance information"""
    chunk_id: str
    doc_id: str
    text: str
    similarity_score: float
    entities: List[str]
    metadata: Dict[str, Any]

class UnifiedSearchPipeline:
    """
    Unified search pipeline that combines the best aspects of the original system
    while fixing the major issues identified.
    """
    
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self.logger = logging.getLogger("UnifiedSearchPipeline")
    
    async def search(
        self,
        question: str,
        conversation_history: Optional[str] = None,
        doc_id: Optional[str] = None,
        scope: SearchScope = SearchScope.HYBRID,
        max_chunks: int = MAX_CHUNKS_PER_ANSWER
    ) -> SearchResult:
        """
        Main search function that implements the improved pipeline
        
        Args:
            question: User's question
            conversation_history: Optional conversation context
            doc_id: Optional document ID to restrict search
            scope: Search scope (global, local, or hybrid)
            max_chunks: Maximum number of chunks to include in context
            
        Returns:
            SearchResult with answer and metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Step 1: Query preprocessing
            processed_question = await self._preprocess_query(question, conversation_history)
            self.logger.info(f"Processed query: {processed_question[:100]}...")
            
            # Step 2: Retrieve relevant chunks using vector similarity
            relevant_chunks = await self._retrieve_relevant_chunks(
                processed_question, doc_id, max_chunks * 2  # Get more candidates for filtering
            )
            self.logger.info(f"Retrieved {len(relevant_chunks)} candidate chunks")
            
            # Step 3: Expand context using graph relationships (optional)
            if scope in [SearchScope.GLOBAL, SearchScope.HYBRID]:
                expanded_chunks = await self._expand_context_via_graph(
                    relevant_chunks, processed_question, max_chunks
                )
                self.logger.info(f"Expanded to {len(expanded_chunks)} chunks via graph")
            else:
                expanded_chunks = relevant_chunks
            
            # Step 4: Rank and filter chunks
            final_chunks = await self._rank_and_filter_chunks(
                expanded_chunks, processed_question, max_chunks
            )
            self.logger.info(f"Final selection: {len(final_chunks)} chunks")
            
            # Step 5: Get relevant community summaries (if available)
            community_summaries = await self._get_relevant_community_summaries(
                processed_question, doc_id, MAX_COMMUNITY_SUMMARIES
            )
            
            # Step 6: Generate answer using selected context
            answer = await self._generate_answer(
                processed_question, final_chunks, community_summaries, conversation_history
            )
            
            # Step 7: Calculate confidence score
            confidence_score = self._calculate_confidence_score(final_chunks, community_summaries)
            
            end_time = asyncio.get_event_loop().time()
            search_time = end_time - start_time
            
            return SearchResult(
                answer=answer,
                relevant_chunks=[chunk.__dict__ for chunk in final_chunks],
                community_summaries=community_summaries,
                entities_found=await self._extract_entities_from_chunks(final_chunks),
                confidence_score=confidence_score,
                search_metadata={
                    "search_time": search_time,
                    "chunks_retrieved": len(relevant_chunks),
                    "chunks_used": len(final_chunks),
                    "scope": scope.value,
                    "doc_id": doc_id
                }
            )
            
        except Exception as e:
            self.logger.error(f"Search pipeline error: {e}")
            return SearchResult(
                answer=f"I apologize, but I encountered an error while searching: {str(e)}",
                relevant_chunks=[],
                community_summaries=[],
                entities_found=[],
                confidence_score=0.0,
                search_metadata={"error": str(e)}
            )
    
    async def _preprocess_query(self, question: str, conversation_history: Optional[str]) -> str:
        """
        Preprocess the query, optionally incorporating conversation history
        """
        if not conversation_history:
            return question.strip()
        
        try:
            # Use LLM to rewrite query with conversation context
            rewrite_prompt = f"""
            Given the conversation history and current question, rewrite the question to be standalone and clear.
            
            Conversation History:
            {conversation_history}
            
            Current Question: {question}
            
            Rewritten Question:"""
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that rewrites questions to be clear and standalone."},
                {"role": "user", "content": rewrite_prompt}
            ]
            
            rewritten = await llm_client.invoke(messages)
            return rewritten.strip() if rewritten.strip() else question
            
        except Exception as e:
            self.logger.warning(f"Query rewriting failed: {e}")
            return question
    
    async def _retrieve_relevant_chunks(
        self, 
        question: str, 
        doc_id: Optional[str], 
        top_k: int
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks using vector similarity search
        """
        try:
            # Get question embedding
            question_embedding = await embedding_client.get_embedding(question)
            
            # Build Cypher query for vector similarity search
            if doc_id:
                cypher = """
                WITH $query_embedding AS queryEmbedding
                MATCH (c:Chunk {doc_id: $doc_id})
                WHERE size(c.embedding) = size(queryEmbedding) AND c.text IS NOT NULL
                WITH c, gds.similarity.cosine(c.embedding, queryEmbedding) AS similarity
                WHERE similarity >= $threshold
                RETURN c.id AS chunk_id, c.doc_id AS doc_id, c.text AS text, 
                       c.document_name AS document_name, similarity
                ORDER BY similarity DESC
                LIMIT $limit
                """
                params = {
                    "query_embedding": question_embedding,
                    "doc_id": doc_id,
                    "threshold": SIMILARITY_THRESHOLD_CHUNKS,
                    "limit": top_k
                }
            else:
                cypher = """
                WITH $query_embedding AS queryEmbedding
                MATCH (c:Chunk)
                WHERE size(c.embedding) = size(queryEmbedding) AND c.text IS NOT NULL
                WITH c, gds.similarity.cosine(c.embedding, queryEmbedding) AS similarity
                WHERE similarity >= $threshold
                RETURN c.id AS chunk_id, c.doc_id AS doc_id, c.text AS text,
                       c.document_name AS document_name, similarity
                ORDER BY similarity DESC
                LIMIT $limit
                """
                params = {
                    "query_embedding": question_embedding,
                    "threshold": SIMILARITY_THRESHOLD_CHUNKS,
                    "limit": top_k
                }
            
            results = await run_cypher_query_async(self.driver, cypher, params)
            
            # Convert to RetrievedChunk objects
            chunks = []
            for result in results:
                # Get entities for this chunk
                entities = await self._get_chunk_entities(result["chunk_id"])
                
                chunks.append(RetrievedChunk(
                    chunk_id=result["chunk_id"],
                    doc_id=result["doc_id"],
                    text=result["text"],
                    similarity_score=result["similarity"],
                    entities=entities,
                    metadata={
                        "document_name": result.get("document_name", ""),
                        "retrieval_method": "vector_similarity"
                    }
                ))
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Chunk retrieval error: {e}")
            return []
    
    async def _get_chunk_entities(self, chunk_id: str) -> List[str]:
        """Get entities associated with a chunk"""
        try:
            cypher = """
            MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk {id: $chunk_id})
            RETURN e.name AS entity_name
            """
            results = await run_cypher_query_async(self.driver, cypher, {"chunk_id": chunk_id})
            return [result["entity_name"] for result in results]
        except Exception as e:
            self.logger.warning(f"Error getting chunk entities: {e}")
            return []
    
    async def _expand_context_via_graph(
        self, 
        initial_chunks: List[RetrievedChunk], 
        question: str, 
        max_total: int
    ) -> List[RetrievedChunk]:
        """
        Expand context by finding related chunks through entity relationships
        """
        if not initial_chunks:
            return initial_chunks
        
        try:
            # Get all entities from initial chunks
            all_entities = set()
            for chunk in initial_chunks:
                all_entities.update(chunk.entities)
            
            if not all_entities:
                return initial_chunks
            
            # Find related chunks through shared entities
            cypher = """
            MATCH (e:Entity)-[:MENTIONED_IN]->(c1:Chunk)
            WHERE e.name IN $entities
            MATCH (e)-[:RELATES_TO]-(related_e:Entity)
            MATCH (related_e)-[:MENTIONED_IN]->(c2:Chunk)
            WHERE c2.id NOT IN $existing_chunk_ids AND c2.text IS NOT NULL
            RETURN DISTINCT c2.id AS chunk_id, c2.doc_id AS doc_id, c2.text AS text,
                   c2.document_name AS document_name, c2.embedding AS embedding
            LIMIT $limit
            """
            
            existing_ids = [chunk.chunk_id for chunk in initial_chunks]
            params = {
                "entities": list(all_entities),
                "existing_chunk_ids": existing_ids,
                "limit": max_total - len(initial_chunks)
            }
            
            results = await run_cypher_query_async(self.driver, cypher, params)
            
            # Calculate similarity scores for related chunks
            question_embedding = await embedding_client.get_embedding(question)
            expanded_chunks = list(initial_chunks)  # Start with initial chunks
            
            for result in results:
                if result.get("embedding"):
                    similarity = cosine_similarity(question_embedding, result["embedding"])
                    if similarity >= SIMILARITY_THRESHOLD_CHUNKS:
                        entities = await self._get_chunk_entities(result["chunk_id"])
                        
                        expanded_chunks.append(RetrievedChunk(
                            chunk_id=result["chunk_id"],
                            doc_id=result["doc_id"],
                            text=result["text"],
                            similarity_score=similarity,
                            entities=entities,
                            metadata={
                                "document_name": result.get("document_name", ""),
                                "retrieval_method": "graph_expansion"
                            }
                        ))
            
            return expanded_chunks
            
        except Exception as e:
            self.logger.warning(f"Graph expansion failed: {e}")
            return initial_chunks
    
    async def _rank_and_filter_chunks(
        self, 
        chunks: List[RetrievedChunk], 
        question: str, 
        max_chunks: int
    ) -> List[RetrievedChunk]:
        """
        Rank chunks by relevance and filter to top results
        """
        if not chunks:
            return []
        
        # Remove duplicates based on chunk_id
        unique_chunks = {}
        for chunk in chunks:
            if chunk.chunk_id not in unique_chunks:
                unique_chunks[chunk.chunk_id] = chunk
            else:
                # Keep the one with higher similarity score
                if chunk.similarity_score > unique_chunks[chunk.chunk_id].similarity_score:
                    unique_chunks[chunk.chunk_id] = chunk
        
        chunks = list(unique_chunks.values())
        
        # Sort by similarity score
        chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Apply relevance threshold
        filtered_chunks = [
            chunk for chunk in chunks 
            if chunk.similarity_score >= RELEVANCE_THRESHOLD
        ]
        
        # Limit to max_chunks
        return filtered_chunks[:max_chunks]
    
    async def _get_relevant_community_summaries(
        self, 
        question: str, 
        doc_id: Optional[str], 
        max_summaries: int
    ) -> List[Dict[str, Any]]:
        """
        Get relevant community summaries based on question similarity
        """
        try:
            # Build query for community summaries
            if doc_id:
                cypher = """
                MATCH (cs:CommunitySummary {doc_id: $doc_id})
                WHERE cs.embedding IS NOT NULL AND cs.summary IS NOT NULL
                RETURN cs.community AS community, cs.summary AS summary, cs.embedding AS embedding
                """
                params = {"doc_id": doc_id}
            else:
                cypher = """
                MATCH (cs:CommunitySummary)
                WHERE cs.embedding IS NOT NULL AND cs.summary IS NOT NULL
                RETURN cs.community AS community, cs.summary AS summary, cs.embedding AS embedding
                LIMIT 20
                """
                params = {}
            
            results = await run_cypher_query_async(self.driver, cypher, params)
            
            if not results:
                return []
            
            # Calculate similarity scores
            question_embedding = await embedding_client.get_embedding(question)
            scored_summaries = []
            
            for result in results:
                if result.get("embedding"):
                    similarity = cosine_similarity(question_embedding, result["embedding"])
                    if similarity >= RELEVANCE_THRESHOLD:
                        scored_summaries.append({
                            "community": result["community"],
                            "summary": result["summary"],
                            "similarity_score": similarity
                        })
            
            # Sort by similarity and return top results
            scored_summaries.sort(key=lambda x: x["similarity_score"], reverse=True)
            return scored_summaries[:max_summaries]
            
        except Exception as e:
            self.logger.warning(f"Community summary retrieval failed: {e}")
            return []
    
    async def _generate_answer(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        community_summaries: List[Dict[str, Any]],
        conversation_history: Optional[str]
    ) -> str:
        """
        Generate answer using retrieved context
        """
        try:
            # Build context string
            context_parts = []
            
            # Add conversation history if available
            if conversation_history:
                context_parts.append(f"Conversation History:\n{conversation_history}\n")
            
            # Add community summaries for high-level context
            if community_summaries:
                context_parts.append("Relevant Topic Summaries:")
                for i, summary in enumerate(community_summaries, 1):
                    context_parts.append(f"{i}. {summary['summary']}")
                context_parts.append("")
            
            # Add specific document excerpts
            if chunks:
                context_parts.append("Relevant Document Excerpts:")
                for i, chunk in enumerate(chunks, 1):
                    # Truncate very long chunks
                    text = chunk.text[:800] + "..." if len(chunk.text) > 800 else chunk.text
                    context_parts.append(f"Excerpt {i} (similarity: {chunk.similarity_score:.2f}): {text}")
                context_parts.append("")
            
            context = "\n".join(context_parts)
            
            # Generate answer
            system_message = """You are a helpful assistant that answers questions based on provided context. 
            Follow these guidelines:
            1. Answer only based on the provided excerpts and summaries
            2. If the answer is not in the context, say you don't know
            3. Be specific and cite relevant information
            4. Keep your answer concise but comprehensive
            5. If multiple sources provide different information, acknowledge this"""
            
            user_message = f"{context}\n\nQuestion: {question}\n\nAnswer:"
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            answer = await llm_client.invoke(messages)
            return answer.strip()
            
        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            return "I apologize, but I encountered an error while generating the answer."
    
    async def _extract_entities_from_chunks(self, chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """Extract and deduplicate entities from retrieved chunks"""
        all_entities = set()
        for chunk in chunks:
            all_entities.update(chunk.entities)
        
        return [{"name": entity, "source": "graph"} for entity in all_entities]
    
    def _calculate_confidence_score(
        self, 
        chunks: List[RetrievedChunk], 
        community_summaries: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate confidence score based on retrieval quality
        """
        if not chunks:
            return 0.0
        
        # Base score from chunk similarities
        avg_similarity = sum(chunk.similarity_score for chunk in chunks) / len(chunks)
        
        # Bonus for having community summaries
        summary_bonus = 0.1 if community_summaries else 0.0
        
        # Bonus for multiple high-quality chunks
        quality_bonus = min(0.2, len([c for c in chunks if c.similarity_score > 0.7]) * 0.05)
        
        confidence = min(1.0, avg_similarity + summary_bonus + quality_bonus)
        return round(confidence, 2)

# Global instance
search_pipeline = None

def get_search_pipeline(neo4j_driver) -> UnifiedSearchPipeline:
    """Get or create the global search pipeline instance"""
    global search_pipeline
    if search_pipeline is None:
        search_pipeline = UnifiedSearchPipeline(neo4j_driver)
    return search_pipeline
