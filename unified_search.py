"""
Unified Search Pipeline for Graph RAG

This module implements a simplified, efficient search pipeline that replaces the complex
multi-endpoint approach with a single, flexible search function. It addresses the key issues:
- Relevant chunk retrieval based on vector similarity
- Proper context filtering and ranking
- Efficient batch processing
- Centralized configuration and structured logging
- Resilience patterns for external service calls (LLM, embeddings, Neo4j)
- Question classification with intelligent routing (BROAD/CHUNK/OUT_OF_SCOPE)
- Map-reduce processing for broad questions

This module uses:
- config.py: Centralized configuration (search parameters, thresholds)
- logging_config.py: Standardized structured logging with context fields
- resilience.py: Automatic retries and circuit breaking for external calls
- question_classifier.py: Intelligent question classification
- map_reduce.py: Map-reduce processing for broad questions
"""

import asyncio
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from neo4j import GraphDatabase

# Import centralized configuration and logging
from config import get_config
from logging_config import get_logger, log_function_call, log_error_with_context
from utils import (
    nlp_processor, llm_client, embedding_client, text_processor,
    cosine_similarity, run_cypher_query_async
)
from question_classifier import classify_question, ClassificationResult
from map_reduce import map_reduce_communities
from mcp_neo4j_cypher import get_mcp_neo4j_client

# Get configuration
cfg = get_config()

# Use configuration values (backward compatibility)
RELEVANCE_THRESHOLD = cfg.search.relevance_threshold
MAX_CHUNKS_PER_ANSWER = cfg.search.max_chunks_per_answer
MAX_COMMUNITY_SUMMARIES = cfg.search.max_community_summaries
SIMILARITY_THRESHOLD_CHUNKS = cfg.search.similarity_threshold_chunks
SIMILARITY_THRESHOLD_ENTITIES = cfg.search.similarity_threshold_entities
EMBEDDING_DIMENSION = cfg.embedding.dimension

# Setup logging using centralized logging config
logger = get_logger("UnifiedSearch")

class SearchScope(Enum):
    """Search scope options"""
    GLOBAL = "global"  # Search across all documents
    LOCAL = "local"    # Search within specific document(s)
    HYBRID = "hybrid"  # Combine global and local approaches

@dataclass
class Citation:
    """Structured citation for API response - grounded to actual source content"""
    citation_id: int
    source_type: str  # "chunk", "community", "cypher_result"
    source_id: str
    source_title: str
    full_text: str  # Complete, untruncated text from database
    relevance_score: float
    metadata: Dict[str, Any] = None  # Additional source metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "citation_id": self.citation_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source_title": self.source_title,
            "full_text": self.full_text,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata or {}
        }

@dataclass
class SearchResult:
    """Structured search result"""
    answer: str
    citations: List[Citation]  # Structured citations with full content
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
    embedding: Optional[List[float]] = None  # For MMR reranking
    final_score: Optional[float] = None  # After reranking

@dataclass
class GraphTraceEdge:
    """Trace record for graph expansion reasoning - captures why a chunk was added during graph expansion"""
    seed_chunk_id: str      # The original chunk that triggered expansion
    seed_entity: str        # Entity in the seed chunk that led to expansion
    related_entity: str     # The related entity found via graph traversal
    added_chunk_id: str     # The new chunk added due to this relationship
    relationship: str       # The relationship type (e.g., "RELATES_TO")


# ============================================================================
# Explainability Helper Functions
# ============================================================================

CITATION_REGEX = r"\[(\d+)\]"


def extract_cited_indices(answer_text: str) -> set:
    """Extract citation indices from LLM answer text.
    
    Handles various citation formats:
    - Simple: [1], [2], [10]
    - Combined: [1,2], [1, 2, 3], [1][2]
    
    Args:
        answer_text: The LLM-generated answer containing citations
        
    Returns:
        Set of integer indices that were cited
    """
    indices = set()
    # Find all bracketed content that might contain numbers
    brackets = re.findall(r'\[([\d,\s]+)\]', answer_text)
    for content in brackets:
        # Extract all numbers from the bracket content
        nums = re.findall(r'\d+', content)
        indices.update(int(n) for n in nums)
    return indices


def sanitize_citations(answer_text: str, max_idx: int) -> str:
    """Replace invalid citation indices with [?].
    
    Args:
        answer_text: The LLM-generated answer
        max_idx: The maximum valid citation index
        
    Returns:
        Answer text with invalid citations replaced by [?]
    """
    def replace_invalid(m):
        idx = int(m.group(1))
        return m.group(0) if 1 <= idx <= max_idx else "[?]"
    return re.sub(CITATION_REGEX, replace_invalid, answer_text)


def validate_citations(answer_text: str, max_idx: int) -> Tuple[str, List[str]]:
    """Validate citations and return warnings for issues.
    
    Args:
        answer_text: The LLM-generated answer
        max_idx: Maximum valid citation index
        
    Returns:
        Tuple of (sanitized answer, list of warning messages)
    """
    warnings = []
    cited = extract_cited_indices(answer_text)
    
    # Check for missing citations
    if not cited:
        warnings.append("⚠️ This answer contains no source citations and may include unsupported statements.")
    
    # Check for out-of-range citations
    invalid = [i for i in cited if i < 1 or i > max_idx]
    if invalid:
        warnings.append(f"⚠️ Some citations could not be verified: {invalid}")
    
    sanitized = sanitize_citations(answer_text, max_idx)
    return sanitized, warnings


def score_to_relevance_label(score: float) -> str:
    """Convert numeric score to user-friendly relevance label."""
    if score >= 0.8:
        return "High relevance"
    elif score >= 0.6:
        return "Medium relevance"
    else:
        return "Low relevance"


def calculate_citation_confidence(answer_text: str, num_sources: int) -> Tuple[float, str]:
    """Calculate confidence based on citation coverage.
    
    Args:
        answer_text: The LLM-generated answer
        num_sources: Number of sources provided
        
    Returns:
        Tuple of (coverage ratio, confidence label)
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+', answer_text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    if not sentences:
        return 0.0, "Low"
    
    # Count sentences with citations
    cited_sentences = sum(1 for s in sentences if re.search(r'\[\d+\]', s))
    coverage = cited_sentences / len(sentences)
    
    if coverage >= 0.7:
        return coverage, "High"
    elif coverage >= 0.4:
        return coverage, "Medium"
    else:
        return coverage, "Low"


# Relation name mapping for readable Graph Trace
RELATION_DISPLAY_MAP = {
    "RELATES_TO": "is related to",
    "MENTIONED_IN": "is mentioned in",
    "CO_OCCURS": "co-occurs with",
    "PART_OF": "is part of",
    "HAS": "has",
    "BELONGS_TO": "belongs to",
    "CONNECTED_TO": "is connected to"
}


def build_explainability_block(
    answer_text: str,
    all_sources: List['RetrievedChunk'],
    graph_trace_edges: List[GraphTraceEdge],
    confidence_score: float,
    explain_cfg
) -> str:
    """Build a markdown explainability block with references and graph trace.
    
    Args:
        answer_text: The LLM-generated answer (used to extract cited indices)
        all_sources: All source chunks in the same order as labeled in prompt
        graph_trace_edges: Graph expansion trace edges
        confidence_score: The calculated confidence score
        explain_cfg: ExplainabilityConfig instance
        
    Returns:
        Markdown-formatted explainability block to append to the answer
    """
    # Validate citations and get warnings
    _, warnings = validate_citations(answer_text, len(all_sources))
    
    # Calculate citation confidence
    coverage, confidence_label = calculate_citation_confidence(answer_text, len(all_sources))
    
    # Extract cited indices
    cited = extract_cited_indices(answer_text)
    
    # Select sources to show
    if explain_cfg.only_cited_sources and cited:
        picked = [(i, all_sources[i-1]) for i in sorted(cited) if 1 <= i <= len(all_sources)]
    else:
        # Fallback: top K sources by their original order
        picked = [(i+1, src) for i, src in enumerate(all_sources[:explain_cfg.max_sources])]
    
    # Format references with improved formatting
    refs_lines = []
    for idx, src in picked:
        doc = src.metadata.get("document_name") or src.metadata.get("doc_id", "Unknown Document")
        score = src.similarity_score
        relevance = score_to_relevance_label(score)
        snippet = src.text[:explain_cfg.snippet_chars].replace("\n", " ").strip()
        if len(src.text) > explain_cfg.snippet_chars:
            snippet += "..."
        refs_lines.append(f"[{idx}] *{doc}* ({relevance})\n    \"{snippet}\"")
    
    # Format graph trace with readable relations
    trace_lines = []
    picked_ids = {src.chunk_id for _, src in picked}
    for edge in graph_trace_edges:
        if edge.added_chunk_id in picked_ids:
            # Use friendly relation name if available
            relation_display = RELATION_DISPLAY_MAP.get(edge.relationship, edge.relationship.lower().replace("_", " "))
            trace_lines.append(
                f"• **{edge.seed_entity}** {relation_display} **{edge.related_entity}**, "
                f"which led to source [{next((i for i, s in picked if s.chunk_id == edge.added_chunk_id), '?')}]"
            )
    
    # Build block with improved formatting
    confidence_detail = f"{confidence_label} – {coverage*100:.0f}% of statements are cited" if cited else "Unable to calculate – no citations found"
    
    block = [
        "\n\n---",
        "## Explainability",
        f"**Confidence:** {confidence_score:.2f} ({confidence_detail})"
    ]
    
    # Add references section
    block.append("\n### References")
    if refs_lines:
        block.append("\n".join(refs_lines))
    else:
        block.append("_No sources were cited in this answer._")
    
    # Add graph trace section if available
    if trace_lines:
        block.append("\n### Graph Trace")
        block.append("\n".join(trace_lines))
    
    # Add warnings if any
    if warnings:
        block.append("\n### Warnings")
        block.append("\n".join(warnings))
    
    return "\n".join(block)


def build_citations_from_chunks(
    answer_text: str,
    all_sources: List['RetrievedChunk']
) -> List[Citation]:
    """Build structured citations from retrieved chunks.
    
    Args:
        answer_text: LLM answer with [n] citations
        all_sources: All source chunks in order
        
    Returns:
        List of Citation objects with full content
    """
    cited = extract_cited_indices(answer_text)
    citations = []
    
    for idx in sorted(cited):
        if 1 <= idx <= len(all_sources):
            src = all_sources[idx - 1]
            citations.append(Citation(
                citation_id=idx,
                source_type="chunk",
                source_id=str(src.chunk_id),  # Ensure source_id is always a string
                source_title=src.metadata.get("document_name") or src.metadata.get("doc_id", "Unknown"),
                full_text=src.text,  # Complete, untruncated
                relevance_score=src.similarity_score,
                metadata={
                    "doc_id": src.doc_id,
                    "entities": src.entities,
                    "retrieval_method": src.metadata.get("retrieval_method", "vector_search")
                }
            ))
    
    return citations


def build_citations_from_communities(
    answer_text: str,
    community_summaries: List[Dict[str, Any]]
) -> List[Citation]:
    """Build structured citations from community summaries.
    
    Args:
        answer_text: LLM answer with [n] citations
        community_summaries: Community summary dicts from Neo4j
        
    Returns:
        List of Citation objects with full content
    """
    cited = extract_cited_indices(answer_text)
    citations = []
    
    # If no citations found in answer, include all communities that were used
    if not cited and community_summaries:
        cited = set(range(1, min(len(community_summaries), 10) + 1))
    
    for idx in sorted(cited):
        if 1 <= idx <= len(community_summaries):
            comm = community_summaries[idx - 1]
            # Use correct key names: 'community' and 'similarity_score'
            community_id = comm.get("community") or comm.get("community_id", f"community_{idx}")
            score = comm.get("similarity_score") or comm.get("score", 0.0)
            
            citations.append(Citation(
                citation_id=idx,
                source_type="community",
                source_id=f"community_{community_id}",
                source_title=f"Community {community_id}",
                full_text=comm.get("summary", ""),  # Complete summary from DB
                relevance_score=score,
                metadata={
                    "entity_count": comm.get("entity_count", 0),
                    "level": comm.get("level", 0)
                }
            ))
    
    return citations


def build_citations_from_cypher(
    query_results: List[Dict[str, Any]],
    metadata: Dict[str, Any]
) -> List[Citation]:
    """Build structured citations from Cypher query results.
    
    Args:
        query_results: Results from Cypher queries
        metadata: Query metadata with strategies
        
    Returns:
        List of Citation objects
    """
    citations = []
    
    if not query_results:
        return citations
    
    for idx, result in enumerate(query_results[:20], 1):  # Limit to 20
        # Build full text from result fields
        full_text_parts = []
        for k, v in result.items():
            if k != "_strategy" and v is not None:
                full_text_parts.append(f"{k}: {v}")
        
        citations.append(Citation(
            citation_id=idx,
            source_type="cypher_result",
            source_id=f"result_{idx}",
            source_title=result.get("entity", result.get("name", f"Result {idx}")),
            full_text="\n".join(full_text_parts),
            relevance_score=0.8,  # Default for Cypher results
            metadata={
                "strategy": result.get("_strategy", "cypher_query"),
                "raw_result": result
            }
        ))
    
    return citations


class UnifiedSearchPipeline:
    """
    Unified search pipeline that combines the best aspects of the original system
    while fixing the major issues identified.
    """
    
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self.logger = get_logger("UnifiedSearchPipeline")
    
    async def search(
        self,
        question: str,
        conversation_history: Optional[str] = None,
        doc_id: Optional[str] = None,
        scope: SearchScope = SearchScope.HYBRID,
        max_chunks: int = MAX_CHUNKS_PER_ANSWER
    ) -> SearchResult:
        """
        Main search function with agentic classification and routing
        
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
            # Step 0: Classify question to determine search strategy
            classification = await classify_question(question)
            question_type = classification["type"]
            classification_reason = classification.get("reason", "")
            classification_confidence = classification.get("confidence", 0.5)
            
            self.logger.info(
                f"Question classified as: {question_type} "
                f"(confidence: {classification_confidence:.2f}, reason: {classification_reason})",
                extra={
                    "question_type": question_type,
                    "confidence": classification_confidence,
                    "reason": classification_reason
                }
            )
            
            # Route based on classification
            if question_type == "OUT_OF_SCOPE":
                return await self._handle_out_of_scope_question(question, start_time)
            elif question_type == "GRAPH_ANALYTICAL":
                return await self._search_with_cypher(
                    question, conversation_history, start_time
                )
            elif question_type == "BROAD":
                return await self._search_with_communities(
                    question, conversation_history, doc_id, max_chunks, start_time
                )
            else:  # CHUNK or default
                return await self._search_with_chunks(
                    question, conversation_history, doc_id, scope, max_chunks, start_time
                )
            
        except Exception as e:
            self.logger.error(f"Search pipeline error: {e}")
            end_time = asyncio.get_event_loop().time()
            return SearchResult(
                answer=f"I apologize, but I encountered an error while searching: {str(e)}",
                relevant_chunks=[],
                community_summaries=[],
                entities_found=[],
                confidence_score=0.0,
                search_metadata={
                    "error": str(e),
                    "search_time": end_time - start_time
                }
            )
    
    async def _handle_out_of_scope_question(self, question: str, start_time: float) -> SearchResult:
        """Handle out-of-scope questions"""
        end_time = asyncio.get_event_loop().time()
        self.logger.info("Question is out of scope")
        
        return SearchResult(
            answer="I'm not confident this question can be answered from the knowledge base. "
                   "Please try rephrasing your question or asking about content in your uploaded documents.",
            relevant_chunks=[],
            community_summaries=[],
            entities_found=[],
            confidence_score=0.0,
            search_metadata={
                "question_type": "OUT_OF_SCOPE",
                "search_time": end_time - start_time,
                "routed": True
            }
        )
    
    async def _search_with_cypher(
        self,
        question: str,
        conversation_history: Optional[str],
        start_time: float
    ) -> SearchResult:
        """
        Search using Neo4j Cypher queries via MCP for graph/analytical questions.
        """
        try:
            self.logger.info("Routing to Cypher-based graph query")
            
            # Check if MCP Neo4j is enabled
            if not cfg.mcp_neo4j.enabled:
                self.logger.warning("MCP Neo4j is disabled, falling back to standard search")
                return await self._search_with_chunks(
                    question, conversation_history, None, SearchScope.GLOBAL, 
                    MAX_CHUNKS_PER_ANSWER, start_time
                )
            
            # Get MCP Neo4j client (pass driver for fallback)
            mcp_client = get_mcp_neo4j_client(neo4j_driver=self.driver)
            
            # Answer question using Cypher
            answer, query_results, metadata = await mcp_client.answer_with_cypher(question)
            
            # Extract entities from results if available
            entities_found = []
            if query_results:
                # Try to extract entity names from results
                for result in query_results[:10]:  # Limit to first 10 results
                    for key, value in result.items():
                        if isinstance(value, str) and len(value) < 100:
                            entities_found.append({"name": value, "source": "cypher_result"})
            
            # Calculate confidence based on whether we got results
            if query_results and len(query_results) > 0:
                confidence_score = 0.8  # High confidence for successful Cypher query
            else:
                confidence_score = 0.3
            
            # Apply explainability for Cypher queries
            if cfg.explainability.enabled:
                # Calculate citation confidence for the answer
                coverage, confidence_label = calculate_citation_confidence(answer, len(query_results) if query_results else 0)
                confidence_detail = f"{confidence_label} – {coverage*100:.0f}% of statements are cited" if coverage > 0 else "Graph query results"
                
                explain_block = [
                    "\n\n---",
                    "## Explainability",
                    f"**Confidence:** {confidence_score:.2f} ({confidence_detail})"
                ]
                
                # Show extracted terms and fuzzy matches
                extracted_terms = metadata.get("extracted_terms", [])
                fuzzy_matches = metadata.get("fuzzy_matches", {})
                
                if extracted_terms:
                    explain_block.append("\n### Terms Extracted")
                    explain_block.append(f"From your question: **{', '.join(extracted_terms)}**")
                
                if fuzzy_matches:
                    fuzzy_info = []
                    for term, matches in fuzzy_matches.items():
                        if matches:
                            fuzzy_info.append(f"'{term}' → '{matches[0]}'")
                    if fuzzy_info:
                        explain_block.append("\n### Fuzzy Entity Matching")
                        explain_block.append(", ".join(fuzzy_info))
                
                # Show query strategies tried
                query_strategies = metadata.get("query_strategies", [])
                if query_strategies:
                    explain_block.append("\n### Query Strategies")
                    for qs in query_strategies:
                        status = "✓" if qs.get("result_count", 0) > 0 else "✗"
                        count = qs.get("result_count", 0)
                        explain_block.append(f"{status} **{qs['type']}**: {qs['description']} ({count} results)")
                
                # Show final query with improved formatting
                final_query = metadata.get("final_query", "")
                if final_query:
                    query_display = final_query[:300] + "..." if len(final_query) > 300 else final_query
                    explain_block.append(f"\n### Best Query\n```cypher\n{query_display}\n```")
                
                explain_block.append(f"\n**Total Results:** {len(query_results) if query_results else 0}")
                
                # Show sample of results with improved formatting
                if query_results and len(query_results) > 0:
                    explain_block.append("\n### Sample Results")
                    for i, result in enumerate(query_results[:5], 1):
                        result_str = ", ".join(f"{k}={v}" for k, v in result.items() 
                                              if isinstance(v, str) and k != "_strategy")[:cfg.explainability.snippet_chars]
                        explain_block.append(f"[{i}] {result_str}")
                
                answer += "\n".join(explain_block)
            
            end_time = asyncio.get_event_loop().time()
            search_time = end_time - start_time
            
            # Build structured citations from Cypher results
            citations = build_citations_from_cypher(query_results, metadata)
            
            return SearchResult(
                answer=answer,
                citations=citations,
                relevant_chunks=[],  # No chunks for Cypher queries
                community_summaries=[],
                entities_found=entities_found[:20],  # Limit entities
                confidence_score=round(confidence_score, 2),
                search_metadata={
                    "question_type": "GRAPH_ANALYTICAL",
                    "search_time": search_time,
                    "strategy": "multi_cypher_query",
                    "iterations": metadata.get("iterations", 0),
                    "final_query": metadata.get("final_query"),
                    "results_count": len(query_results) if query_results else 0,
                    "extracted_terms": metadata.get("extracted_terms", []),
                    "fuzzy_matches": metadata.get("fuzzy_matches", {}),
                    "query_strategies": [{"type": q["type"], "results": q.get("result_count", 0)} 
                                        for q in metadata.get("query_strategies", [])],
                    "explainability_enabled": cfg.explainability.enabled
                }
            )
            
        except Exception as e:
            self.logger.error(f"Cypher search error: {e}")
            end_time = asyncio.get_event_loop().time()
            
            # Fallback to standard search on error
            self.logger.info("Falling back to standard chunk search")
            return await self._search_with_chunks(
                question, conversation_history, None, SearchScope.GLOBAL,
                MAX_CHUNKS_PER_ANSWER, start_time
            )
    
    async def _search_with_communities(
        self,
        question: str,
        conversation_history: Optional[str],
        doc_id: Optional[str],
        max_communities: int,
        start_time: float
    ) -> SearchResult:
        """
        Search using community summaries with map-reduce for broad questions.
        """
        try:
            # Preprocess query
            processed_question = await self._preprocess_query(question, conversation_history)
            
            # Get more communities for map-reduce (default 20, but configurable)
            max_comm = int(os.getenv("BROAD_SEARCH_MAX_COMMUNITIES", "20"))
            community_summaries = await self._get_relevant_community_summaries(
                processed_question, doc_id, max_comm
            )
            
            self.logger.info(f"Retrieved {len(community_summaries)} community summaries for broad search")
            
            # Use map-reduce to generate answer from communities
            answer = await map_reduce_communities(
                processed_question, community_summaries, conversation_history
            )
            
            # Extract entities from communities (optional)
            entities_found = []
            if community_summaries:
                # Try to extract entities from summaries
                all_summaries_text = " ".join([cs.get("summary", "") for cs in community_summaries])
                if all_summaries_text:
                    entities = await nlp_processor.extract_entities(all_summaries_text[:2000])
                    entities_found = [{"name": e.name, "type": e.type} for e in entities]
            
            # Calculate confidence based on community relevance
            if community_summaries:
                avg_similarity = sum(
                    cs.get("similarity_score", 0) for cs in community_summaries
                ) / len(community_summaries)
                confidence_score = min(1.0, avg_similarity + 0.1)  # Boost for broad understanding
            else:
                confidence_score = 0.3
            
            # Apply explainability for broad search (simplified - detail in citations array)
            if cfg.explainability.enabled and community_summaries:
                # Get top scores for display
                top_scores = sorted(
                    [cs.get("similarity_score", 0.0) for cs in community_summaries[:8]],
                    reverse=True
                )[:3]
                avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
                
                explain_block = [
                    "\n\n---",
                    "## Explainability",
                    f"**Confidence:** {confidence_score:.2f}",
                    f"\n**Sources:** {len(community_summaries)} community summaries analyzed",
                    f"**Top relevance scores:** {', '.join(f'{s:.2f}' for s in top_scores)}",
                    "\n_See `citations` array in response for full source content._"
                ]
                
                answer += "\n".join(explain_block)
            
            end_time = asyncio.get_event_loop().time()
            search_time = end_time - start_time
            
            # Build structured citations from community summaries
            citations = build_citations_from_communities(answer, community_summaries)
            
            return SearchResult(
                answer=answer,
                citations=citations,
                relevant_chunks=[],  # No chunks for broad search
                community_summaries=community_summaries,
                entities_found=entities_found,
                confidence_score=round(confidence_score, 2),
                search_metadata={
                    "question_type": "BROAD",
                    "search_time": search_time,
                    "communities_used": len(community_summaries),
                    "strategy": "map_reduce_communities",
                    "doc_id": doc_id,
                    "explainability_enabled": cfg.explainability.enabled
                }
            )
            
        except Exception as e:
            self.logger.error(f"Broad search error: {e}")
            end_time = asyncio.get_event_loop().time()
            return SearchResult(
                answer=f"I encountered an error while processing your broad question: {str(e)}",
                citations=[],
                relevant_chunks=[],
                community_summaries=[],
                entities_found=[],
                confidence_score=0.0,
                search_metadata={
                    "question_type": "BROAD",
                    "error": str(e),
                    "search_time": end_time - start_time
                }
            )
    
    async def _search_with_chunks(
        self,
        question: str,
        conversation_history: Optional[str],
        doc_id: Optional[str],
        scope: SearchScope,
        max_chunks: int,
        start_time: float
    ) -> SearchResult:
        """
        Search using chunk-level retrieval for specific questions.
        This is the original search method, now used for CHUNK-type questions.
        """
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
            trace_edges: List[GraphTraceEdge] = []
            if scope in [SearchScope.GLOBAL, SearchScope.HYBRID]:
                expanded_chunks, trace_edges = await self._expand_context_via_graph(
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
            
            # Step 5: Get relevant community summaries (optional, fewer for chunk search)
            community_summaries = await self._get_relevant_community_summaries(
                processed_question, doc_id, min(3, MAX_COMMUNITY_SUMMARIES)  # Fewer summaries for chunk search
            )
            
            # Step 6: Generate answer using selected context
            answer = await self._generate_answer(
                processed_question, final_chunks, community_summaries, conversation_history
            )
            
            # Step 7: Calculate confidence score
            confidence_score = self._calculate_confidence_score(final_chunks, community_summaries)
            
            # Step 8: Apply explainability if enabled
            if cfg.explainability.enabled and final_chunks:
                # Sanitize any invalid citations
                answer = sanitize_citations(answer, len(final_chunks))
                
                # Append explainability block
                answer += build_explainability_block(
                    answer, final_chunks, trace_edges, 
                    confidence_score, cfg.explainability
                )
            
            end_time = asyncio.get_event_loop().time()
            search_time = end_time - start_time
            
            # Build structured citations from chunks
            citations = build_citations_from_chunks(answer, final_chunks)
            
            return SearchResult(
                answer=answer,
                citations=citations,
                relevant_chunks=[chunk.__dict__ for chunk in final_chunks],
                community_summaries=community_summaries,
                entities_found=await self._extract_entities_from_chunks(final_chunks),
                confidence_score=confidence_score,
                search_metadata={
                    "question_type": "CHUNK",
                    "search_time": search_time,
                    "chunks_retrieved": len(relevant_chunks),
                    "chunks_used": len(final_chunks),
                    "scope": scope.value,
                    "doc_id": doc_id,
                    "strategy": "chunk_level_retrieval",
                    "explainability_enabled": cfg.explainability.enabled
                }
            )
            
        except Exception as e:
            self.logger.error(f"Chunk search error: {e}")
            end_time = asyncio.get_event_loop().time()
            return SearchResult(
                answer=f"I encountered an error while searching: {str(e)}",
                citations=[],
                relevant_chunks=[],
                community_summaries=[],
                entities_found=[],
                confidence_score=0.0,
                search_metadata={
                    "question_type": "CHUNK",
                    "error": str(e),
                    "search_time": end_time - start_time
                }
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
            question_embedding = await embedding_client.get_embedding(question)

            # Primary search: respect doc_id filter when provided
            vector_results = await self._run_vector_similarity_search(
                question_embedding, doc_id, top_k
            )
            chunks = await self._convert_results_to_chunks(
                vector_results, retrieval_method="vector_similarity"
            )

            # Fallback: if doc_id was provided but no chunks were found, retry globally
            if doc_id and not chunks:
                self.logger.info(
                    "No chunks found for specified doc_id; retrying without document filter"
                )
                vector_results = await self._run_vector_similarity_search(
                    question_embedding, None, top_k
                )
                chunks = await self._convert_results_to_chunks(
                    vector_results, retrieval_method="vector_similarity"
                )

            # Secondary fallback: keyword search when vector search yields nothing
            if not chunks:
                chunks = await self._keyword_fallback_search(question, doc_id, top_k)

            return chunks

        except Exception as e:
            self.logger.error(f"Chunk retrieval error: {e}")
            return []

    async def _run_vector_similarity_search(
        self,
        question_embedding: List[float],
        doc_id: Optional[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Execute a vector similarity search using Neo4j native vector search."""
        
        # Use native vector search with db.index.vector.queryNodes
        # This is much faster than computing similarity for all nodes
        try:
            if doc_id:
                cypher = """
                CALL db.index.vector.queryNodes('chunk_embedding_index', $top_k, $query_embedding)
                YIELD node, score
                WHERE node.doc_id = $doc_id AND node.text IS NOT NULL
                RETURN node.id AS chunk_id, node.doc_id AS doc_id, node.text AS text,
                       node.document_name AS document_name, node.embedding AS embedding, score AS similarity
                ORDER BY similarity DESC
                LIMIT $limit
                """
            else:
                cypher = """
                CALL db.index.vector.queryNodes('chunk_embedding_index', $top_k, $query_embedding)
                YIELD node, score
                WHERE node.text IS NOT NULL
                RETURN node.id AS chunk_id, node.doc_id AS doc_id, node.text AS text,
                       node.document_name AS document_name, node.embedding AS embedding, score AS similarity
                ORDER BY similarity DESC
                LIMIT $limit
                """
            
            params = {
                "query_embedding": question_embedding,
                "top_k": top_k * 2,  # Get more candidates for filtering
                "limit": top_k,
                "doc_id": doc_id if doc_id else None
            }
            
            results = await run_cypher_query_async(self.driver, cypher, params)
            
            # Filter by threshold
            filtered_results = [
                r for r in results 
                if r.get("similarity", 0) >= SIMILARITY_THRESHOLD_CHUNKS
            ]
            
            return filtered_results[:top_k]
            
        except Exception as e:
            # Fallback to GDS similarity if vector index doesn't exist
            logger.warning(f"Vector index search failed, falling back to GDS similarity: {e}")
            return await self._run_vector_similarity_search_fallback(question_embedding, doc_id, top_k)
    
    async def _run_vector_similarity_search_fallback(
        self,
        question_embedding: List[float],
        doc_id: Optional[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Fallback to GDS similarity if vector indexes are not available."""
        
        if doc_id:
            cypher = """
            WITH $query_embedding AS queryEmbedding
            MATCH (c:Chunk {doc_id: $doc_id})
            WHERE size(c.embedding) = size(queryEmbedding) AND c.text IS NOT NULL
            WITH c, gds.similarity.cosine(c.embedding, queryEmbedding) AS similarity
            WHERE similarity >= $threshold
            RETURN c.id AS chunk_id, c.doc_id AS doc_id, c.text AS text,
                   c.document_name AS document_name, c.embedding AS embedding, similarity
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
                   c.document_name AS document_name, c.embedding AS embedding, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """
            params = {
                "query_embedding": question_embedding,
                "threshold": SIMILARITY_THRESHOLD_CHUNKS,
                "limit": top_k
            }

        return await run_cypher_query_async(self.driver, cypher, params)

    async def _convert_results_to_chunks(
        self,
        results: List[Dict[str, Any]],
        retrieval_method: str,
        default_similarity: float = SIMILARITY_THRESHOLD_CHUNKS
    ) -> List[RetrievedChunk]:
        """Convert raw cypher results to RetrievedChunk objects."""

        chunks: List[RetrievedChunk] = []

        for result in results:
            entities = await self._get_chunk_entities(result["chunk_id"])
            similarity_score = result.get("similarity", default_similarity)
            
            # Get embedding for MMR reranking
            embedding = result.get("embedding")
            if not embedding:
                # Fetch embedding from database if not in result
                try:
                    cypher = """
                    MATCH (c:Chunk {id: $chunk_id})
                    RETURN c.embedding AS embedding
                    """
                    emb_result = await run_cypher_query_async(
                        self.driver, cypher, {"chunk_id": result["chunk_id"]}
                    )
                    if emb_result:
                        embedding = emb_result[0].get("embedding")
                except Exception as e:
                    self.logger.warning(f"Could not fetch embedding for chunk {result['chunk_id']}: {e}")

            chunks.append(RetrievedChunk(
                chunk_id=result["chunk_id"],
                doc_id=result["doc_id"],
                text=result["text"],
                similarity_score=similarity_score,
                entities=entities,
                embedding=embedding,
                metadata={
                    "document_name": result.get("document_name", ""),
                    "retrieval_method": retrieval_method
                }
            ))

        return chunks

    async def _keyword_fallback_search(
        self,
        question: str,
        doc_id: Optional[str],
        top_k: int
    ) -> List[RetrievedChunk]:
        """Fallback keyword search to avoid empty answers when vector search fails."""

        keywords = [
            kw.lower()
            for kw in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", question)
            if len(kw) > 2
        ][:10]

        if not keywords:
            return []

        doc_filter = "" if not doc_id else "AND c.doc_id = $doc_id"

        cypher = f"""
        UNWIND $keywords AS kw
        MATCH (c:Chunk)
        WHERE c.text IS NOT NULL {doc_filter}
          AND toLower(c.text) CONTAINS kw
        RETURN c.id AS chunk_id, c.doc_id AS doc_id, c.text AS text,
               c.document_name AS document_name, count(kw) AS keyword_matches
        ORDER BY keyword_matches DESC
        LIMIT $limit
        """

        params = {
            "keywords": keywords,
            "limit": top_k,
            "doc_id": doc_id
        }

        try:
            results = await run_cypher_query_async(self.driver, cypher, params)

            if not results and doc_id:
                # Retry globally if doc filter provided nothing
                params["doc_id"] = None
                cypher = cypher.replace("AND c.doc_id = $doc_id", "")
                results = await run_cypher_query_async(self.driver, cypher, params)

            keyword_chunks: List[RetrievedChunk] = []

            for result in results:
                entities = await self._get_chunk_entities(result["chunk_id"])
                match_ratio = result.get("keyword_matches", 0) / max(1, len(keywords))
                similarity_score = max(
                    RELEVANCE_THRESHOLD,
                    round(0.4 + 0.6 * match_ratio, 3)
                )

                keyword_chunks.append(RetrievedChunk(
                    chunk_id=result["chunk_id"],
                    doc_id=result["doc_id"],
                    text=result["text"],
                    similarity_score=similarity_score,
                    entities=entities,
                    metadata={
                        "document_name": result.get("document_name", ""),
                        "retrieval_method": "keyword_fallback"
                    }
                ))

            return keyword_chunks

        except Exception as e:
            self.logger.warning(f"Keyword fallback search failed: {e}")
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
    ) -> Tuple[List[RetrievedChunk], List[GraphTraceEdge]]:
        """
        Expand context by finding related chunks through entity relationships.
        Also captures graph trace edges for explainability.
        
        Returns:
            Tuple of (expanded_chunks, trace_edges)
        """
        trace_edges: List[GraphTraceEdge] = []
        
        if not initial_chunks:
            return initial_chunks, trace_edges
        
        try:
            # Build a mapping from entity to seed chunks that contain it
            entity_to_seed_chunks: Dict[str, List[str]] = {}
            for chunk in initial_chunks:
                for entity in chunk.entities:
                    if entity not in entity_to_seed_chunks:
                        entity_to_seed_chunks[entity] = []
                    entity_to_seed_chunks[entity].append(chunk.chunk_id)
            
            all_entities = set(entity_to_seed_chunks.keys())
            
            if not all_entities:
                return initial_chunks, trace_edges
            
            # Find related chunks through shared entities - include entity info for tracing
            cypher = """
            MATCH (e:Entity)-[:MENTIONED_IN]->(c1:Chunk)
            WHERE e.name IN $entities
            MATCH (e)-[:RELATES_TO]-(related_e:Entity)
            MATCH (related_e)-[:MENTIONED_IN]->(c2:Chunk)
            WHERE c2.id NOT IN $existing_chunk_ids AND c2.text IS NOT NULL
            RETURN DISTINCT c2.id AS chunk_id, c2.doc_id AS doc_id, c2.text AS text,
                   c2.document_name AS document_name, c2.embedding AS embedding,
                   e.name AS seed_entity, related_e.name AS related_entity
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
                        
                        new_chunk = RetrievedChunk(
                            chunk_id=result["chunk_id"],
                            doc_id=result["doc_id"],
                            text=result["text"],
                            similarity_score=similarity,
                            entities=entities,
                            metadata={
                                "document_name": result.get("document_name", ""),
                                "retrieval_method": "graph_expansion"
                            }
                        )
                        expanded_chunks.append(new_chunk)
                        
                        # Create trace edge for explainability
                        seed_entity = result.get("seed_entity", "")
                        related_entity = result.get("related_entity", "")
                        
                        # Find a seed chunk that contains this entity
                        seed_chunk_ids = entity_to_seed_chunks.get(seed_entity, [])
                        seed_chunk_id = seed_chunk_ids[0] if seed_chunk_ids else "unknown"
                        
                        trace_edges.append(GraphTraceEdge(
                            seed_chunk_id=seed_chunk_id,
                            seed_entity=seed_entity,
                            related_entity=related_entity,
                            added_chunk_id=result["chunk_id"],
                            relationship="RELATES_TO"
                        ))
            
            return expanded_chunks, trace_edges
            
        except Exception as e:
            self.logger.warning(f"Graph expansion failed: {e}")
            return initial_chunks, trace_edges
    
    async def _rank_and_filter_chunks(
        self, 
        chunks: List[RetrievedChunk], 
        question: str, 
        max_chunks: int
    ) -> List[RetrievedChunk]:
        """
        Rank chunks using graph-aware reranking and MMR diversity reranking
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
        
        # Apply relevance threshold first
        filtered_chunks = [
            chunk for chunk in chunks 
            if chunk.similarity_score >= RELEVANCE_THRESHOLD
        ]
        
        if not filtered_chunks:
            return []
        
        # Step 1: Graph-aware reranking
        graph_reranked = await self._graph_aware_rerank(filtered_chunks, question)
        
        # Step 2: MMR diversity-aware reranking
        final_chunks = await self._mmr_rerank(graph_reranked, max_chunks, lambda_param=0.7)
        
        return final_chunks
    
    async def _graph_aware_rerank(
        self,
        chunks: List[RetrievedChunk],
        question: str
    ) -> List[RetrievedChunk]:
        """
        Graph-aware reranking using entity overlap, centrality, and community match
        """
        try:
            # Extract entities from query
            query_entities = await nlp_processor.extract_entities(question)
            query_entity_names = {e.name.lower() for e in query_entities}
            
            # Get community scores for chunks
            community_scores = await self._get_community_scores_for_chunks(
                [c.chunk_id for c in chunks], question
            )
            
            # Get centrality scores (precomputed or compute on the fly)
            centrality_scores = await self._get_chunk_centrality_scores(
                [c.chunk_id for c in chunks]
            )
            
            # Get entity overlap for each chunk
            for chunk in chunks:
                chunk_entity_names = {e.lower() for e in chunk.entities}
                entity_overlap = len(query_entity_names & chunk_entity_names)
                
                # Normalize overlap (divide by max possible)
                max_overlap = max(1, len(query_entity_names))
                entity_overlap_score = entity_overlap / max_overlap
                
                # Get community score
                community_score = community_scores.get(chunk.chunk_id, 0.0)
                
                # Get centrality score
                centrality = centrality_scores.get(chunk.chunk_id, 0.0)
                
                # Combine scores: base similarity + graph features
                # Weights: similarity (0.5), entity overlap (0.3), community (0.15), centrality (0.05)
                final_score = (
                    0.5 * chunk.similarity_score +
                    0.3 * entity_overlap_score +
                    0.15 * community_score +
                    0.05 * centrality
                )
                
                chunk.final_score = final_score
                chunk.metadata.update({
                    "entity_overlap": entity_overlap,
                    "entity_overlap_score": entity_overlap_score,
                    "community_score": community_score,
                    "centrality": centrality
                })
            
            # Sort by final score
            chunks.sort(key=lambda x: x.final_score or x.similarity_score, reverse=True)
            return chunks
            
        except Exception as e:
            self.logger.warning(f"Graph-aware reranking failed: {e}, using similarity scores")
            for chunk in chunks:
                chunk.final_score = chunk.similarity_score
            return chunks
    
    async def _get_community_scores_for_chunks(
        self,
        chunk_ids: List[str],
        question: str
    ) -> Dict[str, float]:
        """Get community relevance scores for chunks"""
        try:
            # Get question embedding
            question_embedding = await embedding_client.get_embedding(question)
            
            # Query to get community scores for chunks
            # Find chunks that belong to communities via entities
            cypher = """
            MATCH (c:Chunk)
            WHERE c.id IN $chunk_ids
            MATCH (e:Entity)-[:MENTIONED_IN]->(c)
            MATCH (cs:CommunitySummary)
            WHERE cs.embedding IS NOT NULL AND cs.doc_id = c.doc_id
            WITH c.id AS chunk_id, cs.embedding AS community_embedding, cs.summary AS summary,
                 cs.community AS community
            RETURN DISTINCT chunk_id, community_embedding, summary, community
            """
            
            results = await run_cypher_query_async(
                self.driver, cypher, {"chunk_ids": chunk_ids}
            )
            
            community_scores = {}
            for result in results:
                chunk_id = result["chunk_id"]
                if result.get("community_embedding"):
                    similarity = cosine_similarity(
                        question_embedding, result["community_embedding"]
                    )
                    # Keep max community score for each chunk
                    if chunk_id not in community_scores or similarity > community_scores[chunk_id]:
                        community_scores[chunk_id] = similarity
            
            return community_scores
            
        except Exception as e:
            self.logger.warning(f"Error getting community scores: {e}")
            return {}
    
    async def _get_chunk_centrality_scores(
        self,
        chunk_ids: List[str]
    ) -> Dict[str, float]:
        """Get centrality scores for chunks (precomputed or compute on the fly)"""
        try:
            # Try to get precomputed centrality
            cypher = """
            MATCH (c:Chunk)
            WHERE c.id IN $chunk_ids AND c.centrality IS NOT NULL
            RETURN c.id AS chunk_id, c.centrality AS centrality
            """
            
            results = await run_cypher_query_async(
                self.driver, cypher, {"chunk_ids": chunk_ids}
            )
            
            centrality_scores = {r["chunk_id"]: r["centrality"] for r in results}
            
            # If not precomputed, compute simple degree centrality
            missing_ids = [cid for cid in chunk_ids if cid not in centrality_scores]
            if missing_ids:
                degree_cypher = """
                MATCH (c:Chunk)
                WHERE c.id IN $chunk_ids
                OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(c)
                WITH c.id AS chunk_id, count(e) AS degree
                RETURN chunk_id, degree
                """
                
                degree_results = await run_cypher_query_async(
                    self.driver, degree_cypher, {"chunk_ids": missing_ids}
                )
                
                # Normalize degree centrality
                if degree_results:
                    max_degree = max(r["degree"] for r in degree_results) or 1
                    for result in degree_results:
                        normalized = result["degree"] / max_degree
                        centrality_scores[result["chunk_id"]] = normalized
            
            return centrality_scores
            
        except Exception as e:
            self.logger.warning(f"Error getting centrality scores: {e}")
            return {}
    
    async def _mmr_rerank(
        self,
        candidates: List[RetrievedChunk],
        top_k: int,
        lambda_param: float = 0.7
    ) -> List[RetrievedChunk]:
        """
        Maximal Marginal Relevance (MMR) reranking for diversity
        MMR(doc) = λ * relevance(doc, query) - (1 - λ) * max_sim(doc, already_chosen)
        """
        if not candidates or top_k <= 0:
            return []
        
        # Use final_score if available, otherwise similarity_score
        for c in candidates:
            if c.final_score is None:
                c.final_score = c.similarity_score
        
        selected: List[RetrievedChunk] = []
        remaining = candidates.copy()
        
        while remaining and len(selected) < top_k:
            best_doc = None
            best_score = float('-inf')
            best_idx = -1
            
            for idx, candidate in enumerate(remaining):
                # Relevance score
                relevance = candidate.final_score or candidate.similarity_score
                
                # Diversity penalty: max similarity to already selected chunks
                diversity_penalty = 0.0
                if selected and candidate.embedding:
                    max_similarity = 0.0
                    for selected_chunk in selected:
                        if selected_chunk.embedding:
                            similarity = cosine_similarity(
                                candidate.embedding, selected_chunk.embedding
                            )
                            max_similarity = max(max_similarity, similarity)
                    diversity_penalty = max_similarity
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_doc = candidate
                    best_idx = idx
            
            if best_doc:
                selected.append(best_doc)
                remaining.pop(best_idx)
                best_doc.metadata["mmr_score"] = best_score
            else:
                break
        
        return selected
    
    async def _get_relevant_community_summaries(
        self, 
        question: str, 
        doc_id: Optional[str], 
        max_summaries: int
    ) -> List[Dict[str, Any]]:
        """
        Get relevant community summaries using Neo4j native vector search
        """
        try:
            question_embedding = await embedding_client.get_embedding(question)
            
            # Use native vector search with db.index.vector.queryNodes
            try:
                if doc_id:
                    cypher = """
                    CALL db.index.vector.queryNodes('community_summary_embedding_index', $top_k, $query_embedding)
                    YIELD node, score
                    WHERE node.doc_id = $doc_id AND node.summary IS NOT NULL
                    RETURN node.community AS community, node.summary AS summary, score AS similarity_score
                    ORDER BY similarity_score DESC
                    LIMIT $limit
                    """
                else:
                    cypher = """
                    CALL db.index.vector.queryNodes('community_summary_embedding_index', $top_k, $query_embedding)
                    YIELD node, score
                    WHERE node.summary IS NOT NULL
                    RETURN node.community AS community, node.summary AS summary, score AS similarity_score
                    ORDER BY similarity_score DESC
                    LIMIT $limit
                    """
                
                params = {
                    "query_embedding": question_embedding,
                    "top_k": max_summaries * 2,  # Get more candidates for filtering
                    "limit": max_summaries,
                    "doc_id": doc_id if doc_id else None
                }
                
                results = await run_cypher_query_async(self.driver, cypher, params)
                
                # Filter by threshold
                scored_summaries = [
                    {
                        "community": r["community"],
                        "summary": r["summary"],
                        "similarity_score": r["similarity_score"]
                    }
                    for r in results
                    if r.get("similarity_score", 0) >= RELEVANCE_THRESHOLD
                ]
                
                return scored_summaries[:max_summaries]
                
            except Exception as e:
                # Fallback to cosine similarity if vector index doesn't exist or other error
                error_str = str(e).lower()
                if "dimension" in error_str:
                    self.logger.warning(
                        f"Vector dimension mismatch detected (query: {len(question_embedding)} dims, "
                        f"expected: {EMBEDDING_DIMENSION} dims). "
                        f"Community summaries may have been created with a different embedding model. "
                        f"Using fallback method: {e}"
                    )
                else:
                    self.logger.warning(f"Vector index search for community summaries failed, using fallback: {e}")
                return await self._get_relevant_community_summaries_fallback(question, doc_id, max_summaries)
            
        except Exception as e:
            self.logger.warning(f"Community summary retrieval failed: {e}")
            return []
    
    async def _get_relevant_community_summaries_fallback(
        self, 
        question: str, 
        doc_id: Optional[str], 
        max_summaries: int
    ) -> List[Dict[str, Any]]:
        """Fallback method using cosine similarity in Python"""
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
            query_dim = len(question_embedding)
            expected_dim = EMBEDDING_DIMENSION
            scored_summaries = []
            
            for result in results:
                if result.get("embedding"):
                    stored_embedding = result["embedding"]
                    stored_dim = len(stored_embedding) if stored_embedding else 0
                    
                    # Skip if dimensions don't match the configured dimension
                    if stored_dim != expected_dim or query_dim != expected_dim:
                        self.logger.debug(
                            f"Skipping community summary {result.get('community')} due to dimension mismatch "
                            f"(query: {query_dim}, stored: {stored_dim}, expected: {expected_dim})"
                        )
                        continue
                    
                    similarity = cosine_similarity(question_embedding, stored_embedding)
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
            self.logger.warning(f"Community summary fallback retrieval failed: {e}")
            return []
    
    async def _generate_answer(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        community_summaries: List[Dict[str, Any]],
        conversation_history: Optional[str]
    ) -> str:
        """
        Generate answer using retrieved context.
        When explainability is enabled, sources are labeled [1], [2], etc. and the LLM
        is instructed to cite them inline.
        """
        try:
            # Build context string
            context_parts = []
            
            # Add conversation history if available
            if conversation_history:
                context_parts.append(f"Conversation History:\n{conversation_history}\n")
            
            # Check if explainability is enabled to determine labeling format
            use_citations = cfg.explainability.enabled
            
            # Add specific document excerpts with labeled sources for citations
            if chunks:
                context_parts.append("Sources:")
                for i, chunk in enumerate(chunks, 1):
                    # Truncate very long chunks
                    text = chunk.text[:800] + "..." if len(chunk.text) > 800 else chunk.text
                    doc_name = chunk.metadata.get("document_name", "unknown")
                    
                    if use_citations:
                        # Labeled format for citation support
                        context_parts.append(
                            f"[{i}] (doc={doc_name}, score={chunk.similarity_score:.2f})\n{text}"
                        )
                    else:
                        # Original format
                        context_parts.append(
                            f"Excerpt {i} (similarity: {chunk.similarity_score:.2f}): {text}"
                        )
                context_parts.append("")
            
            # Add community summaries for high-level context
            # Note: for citation purposes, we continue numbering after chunks
            if community_summaries:
                start_idx = len(chunks) + 1 if use_citations else 1
                context_parts.append("Topic Summaries:")
                for i, summary in enumerate(community_summaries):
                    idx = start_idx + i if use_citations else i + 1
                    if use_citations:
                        context_parts.append(f"[{idx}] (community summary)\n{summary['summary']}")
                    else:
                        context_parts.append(f"{idx}. {summary['summary']}")
                context_parts.append("")
            
            context = "\n".join(context_parts)
            
            # Generate answer with citation rules when explainability is enabled
            if use_citations:
                system_message = """You are a helpful assistant that answers questions based on provided context.
You MUST follow these rules:
- Answer ONLY using the provided sources.
- Every non-trivial claim MUST include one or more citations like [1] or [2][5].
- Citations must refer to the provided source indices ONLY.
- If sources do not support the answer, say you don't have enough information.
- Keep your answer concise but comprehensive.
- If multiple sources provide different information, acknowledge this and cite each."""
            else:
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
