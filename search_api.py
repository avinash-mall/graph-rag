from __future__ import annotations
import asyncio
import hashlib
import logging
import os
import json
import random
import re
from typing import List, Dict, Any, Optional, Union, Tuple

import urllib3
import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from neo4j import GraphDatabase
from pydantic import BaseModel

from utils import run_async, clean_text, chunk_text, cosine_similarity, run_cypher_query, get_refined_system_message, \
    extract_entities_with_llm, rerank_community_reports, get_async_llm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

# Global Configuration for search
APP_HOST = os.getenv("APP_HOST") or "0.0.0.0"
APP_PORT = int(os.getenv("APP_PORT") or "8000")
GRAPH_NAME = os.getenv("GRAPH_NAME")
DB_URL = os.getenv("DB_URL")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "llama3.2"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE") or "0.0")
OPENAI_STOP = os.getenv("OPENAI_STOP")
if OPENAI_STOP:
    OPENAI_STOP = json.loads(OPENAI_STOP)
API_TIMEOUT = int(os.getenv("API_TIMEOUT") or "600")

GLOBAL_SEARCH_CHUNK_SIZE = int(os.getenv("GLOBAL_SEARCH_CHUNK_SIZE") or "512")
GLOBAL_SEARCH_TOP_N = int(os.getenv("GLOBAL_SEARCH_TOP_N") or "30")
GLOBAL_SEARCH_BATCH_SIZE = int(os.getenv("GLOBAL_SEARCH_BATCH_SIZE") or "10")
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD") or "0.40")
CYPHER_QUERY_LIMIT = int(os.getenv("CYPHER_QUERY_LIMIT", "5"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("SearchAPI")

# Initialize Neo4j driver
driver = GraphDatabase.driver(DB_URL, auth=(DB_USERNAME, DB_PASSWORD))

# Pydantic models for search endpoints
class QuestionRequest(BaseModel):
    question: str
    doc_id: Optional[Union[str, List[str]]] = None
    previous_conversations: Optional[str] = None

class LocalSearchRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None
    previous_conversations: Optional[str] = None

class DriftSearchRequest(BaseModel):
    question: str
    previous_conversations: Optional[str] = None
    doc_id: Optional[str] = None

class GlobalSearchRequest(BaseModel):
    question: str
    previous_conversations: Optional[str] = None
    doc_id: Optional[str] = None

# ----------------------------
# Asynchronous API Clients (Actual Implementations)
# ----------------------------
class AsyncOpenAI:
    """
    Asynchronous client for interacting with the OpenAI API.
    """
    def __init__(self, api_key: str, model: str, base_url: str, temperature: float, stop: list, timeout: int):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.stop = stop
        self.timeout = timeout

    async def invoke(self, messages: list) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stop": self.stop
        }
        logger.debug(f"LLM Payload Sent: {json.dumps(payload, indent=2)}")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
            response = await client.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return content

    async def invoke_json(self, messages: list, fallback: bool = True) -> str:
        primary_messages = [{"role": "system", "content": get_refined_system_message()}] + messages
        response_text = await self.invoke(primary_messages)
        return response_text

async_llm = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    base_url=OPENAI_BASE_URL,
    temperature=OPENAI_TEMPERATURE,
    stop=OPENAI_STOP,
    timeout=API_TIMEOUT
)

class AsyncEmbeddingAPIClient:
    """
    Asynchronous client for fetching embeddings via the embedding API.
    """
    def __init__(self):
        self.embedding_api_url = os.getenv("EMBEDDING_API_URL", "http://localhost/api/embed")
        self.timeout = API_TIMEOUT
        self.logger = logging.getLogger("EmbeddingAPIClient")
        self.cache = {}

    def _get_text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    async def get_embedding(self, text: str) -> List[float]:
        text_hash = self._get_text_hash(text)
        if text_hash in self.cache:
            self.logger.info(f"Embedding for text (hash: {text_hash}) retrieved from cache.")
            return self.cache[text_hash]
        self.logger.info("Requesting embedding for text (first 50 chars): %s", text[:50])
        payload = {"model": "mxbai-embed-large", "input": text}
        async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
            response = await client.post(
                url=self.embedding_api_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            response_json = response.json()
            embeddings = response_json.get("embeddings", None)
            if not embeddings or not isinstance(embeddings, list) or len(embeddings) == 0:
                raise ValueError(f"Empty or invalid embedding response for text: {text[:50]}")
            embedding = embeddings[0]
            if not embedding:
                raise ValueError("Empty embedding returned for text: " + text[:50])
            self.cache[text_hash] = embedding
            return embedding

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        tasks = [self.get_embedding(text) for text in texts]
        return await asyncio.gather(*tasks)

async_embedding_client = AsyncEmbeddingAPIClient()

# ----------------------------
# Helper Functions for Search
# ----------------------------
async def rewrite_query_if_needed(question: str, conversation_history: Optional[str]) -> str:
    """
    Rewrite a user query into a standalone query if conversation history is provided.
    """
    if conversation_history:
        rewrite_prompt = (
            f"Based on the previous conversation: '{conversation_history}', "
            f"rewrite the query: '{question}' into a standalone query."
        )
        rewritten_query = await async_llm.invoke([
            {"role": "system", "content": "You are a professional query rewriting assistant."},
            {"role": "user", "content": rewrite_prompt}
        ])
        return rewritten_query.strip()
    return question

async def global_search_map_reduce_plain(
        question: str,
        conversation_history: Optional[str] = None,
        doc_id: Optional[str] = None,
        chunk_size: int = GLOBAL_SEARCH_CHUNK_SIZE,
        top_n: int = GLOBAL_SEARCH_TOP_N,
        batch_size: int = GLOBAL_SEARCH_BATCH_SIZE
) -> str:
    # Step 1: Compute the embedding for the user query.
    query_embedding = await async_embedding_client.get_embedding(question)

    # Step 2: Retrieve similar Chunk nodes using cosine similarity.
    cypher = """
    WITH $query_embedding AS queryEmbedding
    MATCH (c:Chunk)
    WHERE size(c.embedding) = size(queryEmbedding)
    WITH c, gds.similarity.cosine(c.embedding, queryEmbedding) AS sim
    WHERE sim > $similarity_threshold
    RETURN c.doc_id AS doc_id, c.text AS chunk_text, sim
    ORDER BY sim DESC
    LIMIT $limit
    """
    params = {
        "query_embedding": query_embedding,
        "similarity_threshold": 0.4,
        "limit": GLOBAL_SEARCH_TOP_N
    }
    similar_chunks = await asyncio.to_thread(run_cypher_query, driver, cypher, params)

    # Step 3: Retrieve community summaries based on the doc_ids from similar chunks.
    related_doc_ids = list({record["doc_id"] for record in similar_chunks})
    community_summaries = {}
    for doc in related_doc_ids:
        query_cs = """
         MATCH (cs:CommunitySummary {doc_id: $doc_id})
         RETURN cs.community AS community, cs.summary AS summary, cs.embedding AS embedding
         """
        cs_result = await asyncio.to_thread(run_cypher_query, driver, query_cs, {"doc_id": doc})
        for record in cs_result:
            community_summaries[record["community"]] = {
                "summary": record["summary"],
                "embedding": record["embedding"]
            }
    if not community_summaries:
        raise HTTPException(status_code=500, detail="No related community summaries found based on chunk similarity.")

    # Step 4: Process community summaries to extract key points.
    # Instead of randomly shuffling, use the LLM to rerank dynamically.
    community_reports = list(community_summaries.values())
    community_reports = await rerank_community_reports(question, community_reports)

    intermediate_points = []
    for report in community_reports:
        chunks = chunk_text(report["summary"], max_chunk_size=chunk_size)
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_prompt = (
                "You are an expert in extracting key points. For the following text chunks from a community summary, "
                "list the key points that are most relevant to answering the user query. For each key point, provide "
                "a brief description and assign a relevance rating between 1 and 100. "
                "Format each key point on a separate line in this format:\n"
                "Key point: <description> (Rating: <number>)\n\n"
            )
            for idx, chunk in enumerate(batch):
                batch_prompt += f"Chunk {idx + 1}:\n\"\"\"\n{chunk}\n\"\"\"\n\n"
            batch_prompt += f"User Query: \"{question}\"\n"
            try:
                response = await async_llm.invoke([
                    {"role": "system", "content": "You are a professional extraction assistant."},
                    {"role": "user", "content": batch_prompt}
                ])
                intermediate_points.append(response.strip())
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
    aggregated_key_points = "\n".join(intermediate_points)

    # Step 4.5: Use LLM prompting to rerank the aggregated key points.
    rerank_prompt = (
        "You are an expert at evaluating the relevance of key points extracted from community summaries. "
        "Below are key points extracted from the documents, each on a separate line. Please rerank them in descending "
        "order of relevance to the following user query, and output the reranked key points as a plain text list, "
        "each on a separate line.\n"
        f"User Query: \"{question}\"\n"
        "Key Points:\n" + aggregated_key_points + "\n"
        "Output the reranked key points in order (most relevant first), preserving their original text.\n"
    )
    try:
        reranked_key_points = await async_llm.invoke([
            {"role": "system", "content": "You are an expert at ranking key points."},
            {"role": "user", "content": rerank_prompt}
        ])
    except Exception as e:
        logger.error(f"Error during key points reranking: {e}")
        reranked_key_points = aggregated_key_points

    conv_text = f"Conversation History: {conversation_history}\n" if conversation_history else ""

    # Step 5: Build a reduction prompt using the reranked key points to generate the final answer.
    reduce_prompt = f"""
{conv_text}You are a professional assistant tasked with synthesizing a detailed answer from the following reranked key points:
{reranked_key_points}

User Query: "{question}"

Using only the above key points, generate a comprehensive and detailed answer in plain text that directly addresses the query. Please include a clear summary and explanation that ties together all key points.
"""
    final_answer = await async_llm.invoke([
        {"role": "system", "content": "You are a professional assistant providing detailed answers."},
        {"role": "user", "content": reduce_prompt}
    ])

    return final_answer

def select_relevant_communities(query: str, community_reports: List[Union[str, dict]], top_k: int = GLOBAL_SEARCH_TOP_N,
                                threshold: float = RELEVANCE_THRESHOLD) -> List[Tuple[str, float]]:
    """
    Select and rank community reports that are relevant to the query using cosine similarity.
    """
    query_embedding = run_async(async_embedding_client.get_embedding(query))
    scored_reports = []
    for report in community_reports:
        if isinstance(report, dict) and "embedding" in report:
            report_embedding = report["embedding"]
            summary_text = report["summary"]
        else:
            summary_text = report
            report_embedding = run_async(async_embedding_client.get_embedding(report))
        score = cosine_similarity(query_embedding, report_embedding)
        logger.info("Computed cosine similarity for community report snippet '%s...' is %.4f", summary_text[:50], score)
        if score >= threshold:
            scored_reports.append((summary_text, score))
        else:
            logger.info("Filtered out community report snippet '%s...' with score %.4f (threshold: %.2f)",
                        summary_text[:50], score, threshold)
    scored_reports.sort(key=lambda x: x[1], reverse=True)
    logger.info("Selected community reports after filtering:")
    for rep, score in scored_reports:
        logger.info("Score: %.4f, Snippet: %s", score, rep[:100])
    return scored_reports[:top_k]

# ----------------------------
# API Router for Search/Answering Endpoints
# ----------------------------
router = APIRouter()


@router.post("/cypher_search")
async def cypher_search(request: QuestionRequest):
    # Step 1: Extract candidate entities from the user query using our common NER function.
    candidate_entities = []
    try:
        entities = await extract_entities_with_llm(request.question)
        candidate_entities = [entity["name"] for entity in entities if entity.get("name", "").strip()]
        if not candidate_entities:
            raise ValueError("No entities found")
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        candidate_entities = request.question.split()[:2]

    final_entities = []
    # For each candidate, compute similarity using embeddings in Neo4j.
    if candidate_entities:
        for candidate in candidate_entities:
            try:
                candidate_embedding = await async_embedding_client.get_embedding(candidate)
            except Exception as e:
                logger.error(f"Error getting embedding for candidate '{candidate}': {e}")
                continue

            cypher_query = """
            WITH $candidate_embedding AS candidateEmbedding
            MATCH (e:Entity)
            WHERE size(e.embedding) = size(candidateEmbedding)
            WITH e, gds.similarity.cosine(e.embedding, candidateEmbedding) AS similarity
            WHERE similarity > 0.8
            RETURN e.name AS name, similarity
            ORDER BY similarity DESC
            LIMIT 1
            """
            try:
                result = await asyncio.to_thread(run_cypher_query, driver, cypher_query,
                                                 {"candidate_embedding": candidate_embedding})
            except Exception as e:
                logger.error(f"Error executing similarity query for candidate '{candidate}': {e}")
                continue
            if result and len(result) > 0:
                top_match = result[0].get("name")
                if top_match:
                    final_entities.append(top_match)
    final_entities = list(set(final_entities))

    results = []
    # New fallback logic if no matching entities were found via NER-based similarity.
    if not final_entities:
        try:
            # Step 1: Compute query embedding.
            query_embedding = await async_embedding_client.get_embedding(request.question)
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            raise HTTPException(status_code=500, detail="Error computing query embedding")

        # Step 2: Perform cosine similarity search against all Entity embeddings.
        FALLBACK_SIMILARITY_THRESHOLD = float(os.getenv("FALLBACK_SIMILARITY_THRESHOLD") or "0.0")
        FALLBACK_ENTITY_LIMIT = int(os.getenv("FALLBACK_ENTITY_LIMIT") or "3")
        fallback_cosine_query = """
            WITH $query_embedding AS queryEmbedding
            MATCH (e:Entity)
            WHERE size(e.embedding) = size(queryEmbedding)
            WITH e, gds.similarity.cosine(e.embedding, queryEmbedding) AS sim
            WHERE sim > $fallback_similarity_threshold
            RETURN e.name AS name, sim
            ORDER BY sim DESC
            LIMIT $fallback_entity_limit
        """
        fallback_params = {
            "query_embedding": query_embedding,
            "fallback_similarity_threshold": FALLBACK_SIMILARITY_THRESHOLD,
            "fallback_entity_limit": FALLBACK_ENTITY_LIMIT
        }
        fallback_entities = await asyncio.to_thread(run_cypher_query, driver, fallback_cosine_query, fallback_params)

        # Step 3: For each top entity, run a JaroWinkler-based query.
        FALLBACK_JARO_THRESHOLD = float(os.getenv("FALLBACK_JARO_THRESHOLD") or "0.8")
        FALLBACK_QUERY_LIMIT = int(os.getenv("FALLBACK_QUERY_LIMIT") or "3")
        for fallback_entity in fallback_entities:
            entity_name = fallback_entity["name"]
            fallback_jaro_query = f"""
                MATCH (e:Entity)
                WHERE apoc.text.jaroWinklerDistance(e.name, "{entity_name}") > $fallback_jaro_threshold
                WITH e
                MATCH (e)-[:RELATES_TO]->(related:Entity)
                OPTIONAL MATCH (related)-[:MENTIONED_IN]->(chunk:Chunk)
                RETURN e.name AS MatchedEntity, related.name AS RelatedEntity, collect(chunk.text) AS ChunkTexts
                LIMIT $fallback_query_limit
            """
            jaro_params = {
                "fallback_jaro_threshold": FALLBACK_JARO_THRESHOLD,
                "fallback_query_limit": FALLBACK_QUERY_LIMIT
            }
            result = await asyncio.to_thread(run_cypher_query, driver, fallback_jaro_query, jaro_params)
            results.append(result)
    else:
        # For each extracted entity, generate three cypher queries.
        for entity in final_entities:
            query_a = f"""
            MATCH (start:Entity {{name: "{entity}"}})
            OPTIONAL MATCH (start)-[:RELATES_TO]->(related:Entity)
            OPTIONAL MATCH (related)-[:MENTIONED_IN]->(chunk:Chunk)
            RETURN start.name AS StartingEntity, related.name AS RelatedEntity, chunk.text AS ChunkText
            LIMIT {CYPHER_QUERY_LIMIT};
            """
            query_b = f"""
            MATCH (start:Entity {{name: "{entity}"}})-[:RELATES_TO]->(related:Entity)
            OPTIONAL MATCH (related)-[:MENTIONED_IN]->(chunk:Chunk)
            RETURN related.name AS RelatedEntity, collect(chunk.text) AS ChunkTexts
            LIMIT {CYPHER_QUERY_LIMIT};
            """
            query_c = f"""
            MATCH (start:Entity {{name: "{entity}"}})
            OPTIONAL MATCH (start)-[:RELATES_TO]->(related:Entity)
            OPTIONAL MATCH (related)-[:MENTIONED_IN]->(chunk:Chunk)
            RETURN start.name AS StartingEntity, related.name AS RelatedEntity, chunk.text AS ChunkText
            LIMIT {CYPHER_QUERY_LIMIT};
            """
            for q in [query_a, query_b, query_c]:
                try:
                    qr = await asyncio.to_thread(run_cypher_query, driver, q)
                    results.append(qr)
                except Exception as e:
                    logger.error(f"Error executing query for entity '{entity}': {e}")

    # Step 4: Aggregate all returned text chunks (map reduce) without duplicates.
    processed_chunks = set()
    aggregated_text = ""
    for res in results:
        for record in res:
            if "ChunkText" in record and record["ChunkText"]:
                chunk = record["ChunkText"].strip()
                if chunk and chunk not in processed_chunks:
                    processed_chunks.add(chunk)
                    aggregated_text += " " + chunk
            elif "ChunkTexts" in record and record["ChunkTexts"]:
                for chunk in record["ChunkTexts"]:
                    chunk = chunk.strip()
                    if chunk and chunk not in processed_chunks:
                        processed_chunks.add(chunk)
                        aggregated_text += " " + chunk
            elif "MatchedEntity" in record and "ChunkTexts" in record:
                if record["ChunkTexts"]:
                    for chunk in record["ChunkTexts"]:
                        chunk = chunk.strip()
                        if chunk and chunk not in processed_chunks:
                            processed_chunks.add(chunk)
                            aggregated_text += " " + chunk
    aggregated_text = aggregated_text.strip()

    # Step 5: Use the aggregated text along with the original query to generate the final answer.
    final_prompt = (
        f"User Query: {request.question}\n\n"
        f"Extracted Graph Data:\n{aggregated_text}\n\n"
        "Based solely on the above information, provide a detailed answer to the query."
    )
    try:
        final_answer = await async_llm.invoke([
            {"role": "system", "content": "You are a professional assistant."},
            {"role": "user", "content": final_prompt}
        ])
    except Exception as e:
        logger.error(f"Error generating final answer: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating final answer: {e}")

    return {
        "final_answer": final_answer,
        "aggregated_text": aggregated_text,
        "executed_queries": results
    }

@router.post("/global_search")
async def global_search(request: GlobalSearchRequest):
    """
    Global search endpoint that rewrites the query (if needed) and synthesizes an answer using key points from community summaries.
    """
    request.question = await rewrite_query_if_needed(request.question, request.previous_conversations)
    final_answer_text = await global_search_map_reduce_plain(
        question=request.question,
        conversation_history=request.previous_conversations,
        doc_id=request.doc_id,
        chunk_size=GLOBAL_SEARCH_CHUNK_SIZE,
        top_n=GLOBAL_SEARCH_TOP_N,
        batch_size=GLOBAL_SEARCH_BATCH_SIZE
    )
    return {"answer": final_answer_text}

@router.post("/local_search")
async def local_search(request: LocalSearchRequest):
    """
    Local search endpoint that uses conversation history and document context to generate an answer.
    """
    request.question = await rewrite_query_if_needed(request.question, request.previous_conversations)
    conversation_context = f"Conversation History:\n{request.previous_conversations}\n\n" if request.previous_conversations else ""
    if request.doc_id:
        query = """
        MATCH (cs:CommunitySummary {doc_id: $doc_id})
        RETURN cs.community AS community, cs.summary AS summary, cs.embedding AS embedding
        """
        summaries_result = await asyncio.to_thread(run_cypher_query, driver, query, {"doc_id": request.doc_id})
        summaries = {record["community"]: {"summary": record["summary"], "embedding": record["embedding"]} for record in summaries_result}
    else:
        query = """
        MATCH (cs:CommunitySummary)
        RETURN cs.community AS community, cs.summary AS summary, cs.embedding AS embedding
        """
        summaries_result = await asyncio.to_thread(run_cypher_query, driver, query)
        summaries = {record["community"]: {"summary": record["summary"], "embedding": record["embedding"]} for record in summaries_result}
    if summaries:
        query_embedding = await async_embedding_client.get_embedding(request.question)
        scored_summaries = []
        for comm, info in summaries.items():
            if info.get("embedding"):
                summary_embedding = info["embedding"]
                score = cosine_similarity(query_embedding, summary_embedding)
                scored_summaries.append((comm, info["summary"], score))
        threshold = 0.3
        filtered_summaries = [(comm, summary, score) for comm, summary, score in scored_summaries if score >= threshold]
        filtered_summaries.sort(key=lambda x: x[2], reverse=True)
        top_k = 3
        top_summaries = filtered_summaries[:top_k]
        community_context = "Community Summaries (ranked by similarity):\n" + "\n".join(
            [f"{comm}: {summary} (Score: {score:.2f})" for comm, summary, score in top_summaries]
        ) + "\n"
    else:
        community_context = "No community summaries available.\n"
    if request.doc_id:
        text_unit_query = """
        MATCH (c:Chunk)
        WHERE c.doc_id = $doc_id
        RETURN c.text AS chunk_text
        LIMIT 5
        """
        text_unit_results = await asyncio.to_thread(run_cypher_query, driver, text_unit_query, {"doc_id": request.doc_id})
    else:
        text_unit_query = """
        MATCH (c:Chunk)
        RETURN c.text AS chunk_text
        LIMIT 5
        """
        text_unit_results = await asyncio.to_thread(run_cypher_query, driver, text_unit_query)
    if text_unit_results:
        text_unit_context = "Document Text Units:\n" + "\n---\n".join(
            [res.get("chunk_text", "") for res in text_unit_results if res.get("chunk_text")]
        ) + "\n"
    else:
        text_unit_context = "No document text units found.\n"
    combined_context = conversation_context + community_context + text_unit_context
    prompt_template = (
        "You are a professional assistant who answers queries strictly based on the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    prompt = prompt_template.format(context=combined_context, question=request.question)
    logger.debug(f"Local search prompt:\n{prompt}")
    answer = await async_llm.invoke([
        {"role": "system", "content": "You are a professional assistant who answers strictly based on the provided context."},
        {"role": "user", "content": prompt}
    ])
    return {"local_search_answer": answer}

@router.post("/drift_search")
async def drift_search(request: DriftSearchRequest):
    """
    Drift search endpoint that refines the initial query with a multi-phase process (global primer, local phase, reduction).
    """
    request.question = await rewrite_query_if_needed(request.question, request.previous_conversations)
    # Global Phase
    query_embedding = await async_embedding_client.get_embedding(request.question)
    cypher = """
    WITH $query_embedding AS queryEmbedding
    MATCH (c:Chunk)
    WHERE size(c.embedding) = size(queryEmbedding)
    WITH c, gds.similarity.cosine(c.embedding, queryEmbedding) AS sim
    WHERE sim > $similarity_threshold
    RETURN c.doc_id AS doc_id, c.text AS chunk_text, sim
    ORDER BY sim DESC
    LIMIT 30
    """
    params = {
        "query_embedding": query_embedding,
        "similarity_threshold": 0.4,
        "limit": 30
    }
    similar_chunks = await asyncio.to_thread(run_cypher_query, driver, cypher, params)
    related_doc_ids = list({record["doc_id"] for record in similar_chunks})
    community_summaries = {}
    for doc in related_doc_ids:
        query_cs = """
         MATCH (cs:CommunitySummary {doc_id: $doc_id})
         RETURN cs.community AS community, cs.summary AS summary, cs.embedding AS embedding
         """
        cs_result = await asyncio.to_thread(run_cypher_query, driver, query_cs, {"doc_id": doc})
        for record in cs_result:
            community_summaries[record["community"]] = {
                "summary": record["summary"],
                "embedding": record["embedding"]
            }
    if not community_summaries:
        raise HTTPException(status_code=500, detail="No related community summaries found based on chunk similarity.")
    community_reports = [v["summary"] for v in community_summaries.values() if v.get("summary")]
    random.shuffle(community_reports)
    # Primer Phase
    primer_prompt = f"""
You are an expert in synthesizing information from diverse community reports.
Based on the following global context derived from document similarity filtering:
{ "\n".join(community_reports) }
{"Conversation History:\n" + request.previous_conversations if request.previous_conversations else ""}
Please provide a preliminary answer to the following query and list any follow-up questions that could help refine it.
Format your response as plain text with the following sections:

Intermediate Answer:
<your intermediate answer here>

Follow-Up Questions:
1. <first follow-up question>
2. <second follow-up question>
... 

Query: {request.question}
"""
    primer_result = await async_llm.invoke([
        {"role": "system", "content": "You are a professional assistant."},
        {"role": "user", "content": primer_prompt}
    ])
    intermediate_answer = ""
    follow_up_questions = []
    if "Intermediate Answer:" in primer_result and "Follow-Up Questions:" in primer_result:
        parts = primer_result.split("Follow-Up Questions:")
        intermediate_part = parts[0]
        follow_up_part = parts[1]
        if "Intermediate Answer:" in intermediate_part:
            intermediate_answer = intermediate_part.split("Intermediate Answer:")[1].strip()
        follow_up_lines = follow_up_part.strip().splitlines()
        for line in follow_up_lines:
            line = line.strip()
            if line:
                if line[0].isdigit():
                    dot_index = line.find('.')
                    if dot_index != -1:
                        line = line[dot_index + 1:].strip()
                follow_up_questions.append(line)
    else:
        intermediate_answer = primer_result.strip()
    drift_hierarchy = {
        "query": request.question,
        "answer": intermediate_answer,
        "follow_ups": []
    }
    # Local Phase
    for follow_up in follow_up_questions:
        local_chunk_query = """
             MATCH (c:Chunk)
             WHERE toLower(c.text) CONTAINS toLower($keyword)
        """
        if request.doc_id:
            local_chunk_query += " AND c.doc_id = $doc_id"
        local_chunk_query += "\nRETURN c.text AS chunk_text\nLIMIT 5"
        params = {"keyword": follow_up}
        if request.doc_id:
            params["doc_id"] = request.doc_id
        chunk_results = await asyncio.to_thread(run_cypher_query, driver, local_chunk_query, params)
        local_context_text = ""
        if chunk_results:
            chunks = [res.get("chunk_text", "") for res in chunk_results if res.get("chunk_text")]
            local_context_text = "Related Document Chunks:\n" + "\n---\n".join(chunks) + "\n"
        local_conversation = f"Conversation History:\n{request.previous_conversations}\n\n" if request.previous_conversations else ""
        local_prompt = f"""
You are a professional assistant who refines queries using local document context.
Based on the following local context:
{local_context_text}
{local_conversation}
Please provide an answer to the following follow-up query and list any additional follow-up questions.
Format your response as plain text with these sections:

Answer:
<your answer here>

Follow-Up Questions:
1. <first follow-up question>
2. <second follow-up question>
...

Follow-Up Query: {follow_up}
"""
        local_result = await async_llm.invoke([
            {"role": "system", "content": "You are a professional assistant."},
            {"role": "user", "content": local_prompt}
        ])
        local_answer = ""
        local_follow_ups = []
        if "Answer:" in local_result and "Follow-Up Questions:" in local_result:
            parts = local_result.split("Follow-Up Questions:")
            answer_part = parts[0]
            follow_up_part = parts[1]
            if "Answer:" in answer_part:
                local_answer = answer_part.split("Answer:")[1].strip()
            follow_up_lines = follow_up_part.strip().splitlines()
            for line in follow_up_lines:
                line = line.strip()
                if line:
                    if line[0].isdigit():
                        dot_index = line.find('.')
                        if dot_index != -1:
                            line = line[dot_index + 1:].strip()
                    local_follow_ups.append(line)
        else:
            local_answer = local_result.strip()
        drift_hierarchy["follow_ups"].append({
            "query": follow_up,
            "answer": local_answer,
            "follow_ups": local_follow_ups
        })
    # Reduction Phase
    reduction_prompt = f"""
You are a professional assistant tasked with synthesizing a final detailed answer.
Below is the hierarchical data gathered:

Query: {drift_hierarchy["query"]}

Intermediate Answer: {drift_hierarchy["answer"]}

Follow-Up Interactions:
"""
    for idx, fu in enumerate(drift_hierarchy["follow_ups"], start=1):
        reduction_prompt += f"\nFollow-Up {idx}:\nQuery: {fu['query']}\nAnswer: {fu['answer']}\n"
        if fu["follow_ups"]:
            reduction_prompt += "Additional Follow-Ups: " + ", ".join(fu["follow_ups"]) + "\n"
    reduction_prompt += f"\nBased solely on the above, provide a final, comprehensive answer in plain text to the original query:\n{request.question}"
    final_answer = await async_llm.invoke([
        {"role": "system", "content": "You are a professional assistant."},
        {"role": "user", "content": reduction_prompt}
    ])
    return {"drift_search_answer": final_answer.strip(), "drift_hierarchy": drift_hierarchy}
