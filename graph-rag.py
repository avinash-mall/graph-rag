from __future__ import annotations
import asyncio
import threading
import hashlib
import json
import logging
import os
import random
import re
import string
import sys
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from lmformatenforcer import JsonSchemaParser
import json
import PyPDF2
import blingfire
import docx
import json5  # More forgiving JSON parser. Install with: pip install json5
import httpx
import urllib3
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from neo4j import GraphDatabase
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Helper function to run asynchronous coroutines in a thread-safe manner.
# This function allows execution of async coroutines from synchronous contexts.
# -----------------------------------------------------------------------------
def run_async(coro):
    """
    Run an asynchronous coroutine safely without calling asyncio.run() within a running loop.
    :param coro: The coroutine to run.
    :return: The result of the coroutine execution.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        result_container = {}
        new_loop = None  # Initialize new_loop outside the inner function.
        def run():
            nonlocal new_loop  # Declare nonlocal to modify the outer new_loop.
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                result_container["result"] = new_loop.run_until_complete(coro)
            except Exception as e:
                result_container["result"] = e
            finally:
                if new_loop:
                    new_loop.close()
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        result = result_container.get("result")
        if isinstance(result, Exception):
            raise result
        return result
    else:
        return asyncio.run(coro)

# -----------------------------------------------------------------------------
# Disable warnings and load environment variables
# -----------------------------------------------------------------------------
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()  # Load variables from the .env file

# -----------------------------------------------------------------------------
# Application & Global Configuration
# -----------------------------------------------------------------------------
APP_HOST = os.getenv("APP_HOST") or "0.0.0.0"  # Host address for the application
APP_PORT = int(os.getenv("APP_PORT") or "8000")  # Port for the application

# Graph & Neo4j Configuration
GRAPH_NAME = os.getenv("GRAPH_NAME")
DB_URL = os.getenv("DB_URL")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE") or "0.0")
OPENAI_STOP = json.loads(os.getenv("OPENAI_STOP") or '["<|end_of_text|>", "<|eot_id|>"]')
OPENAI_API_TIMEOUT = int(os.getenv("API_TIMEOUT") or "600")

# Chunking & Global Search Defaults
CHUNK_SIZE_GDS = int(os.getenv("CHUNK_SIZE_GDS") or "1024")
GLOBAL_SEARCH_CHUNK_SIZE = int(os.getenv("GLOBAL_SEARCH_CHUNK_SIZE") or "1024")
GLOBAL_SEARCH_TOP_N = int(os.getenv("GLOBAL_SEARCH_TOP_N") or "5")
GLOBAL_SEARCH_BATCH_SIZE = int(os.getenv("GLOBAL_SEARCH_BATCH_SIZE") or "20")
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD") or "0.1")
COREF_WORD_LIMIT = int(os.getenv("COREF_WORD_LIMIT", "8000"))
CYPHER_QUERY_LIMIT = int(os.getenv("CYPHER_QUERY_LIMIT", "5"))

# Logging Configuration (optional)
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", "app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, LOG_FILE), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GraphRAG")
logger.info(f"Logging initialized with level: {LOG_LEVEL}")

# -----------------------------------------------------------------------------
# Neo4j Driver Setup: Creates a driver instance for connecting to the Neo4j database.
# -----------------------------------------------------------------------------
driver = GraphDatabase.driver(DB_URL, auth=(DB_USERNAME, DB_PASSWORD))


# -----------------------------------------------------------------------------
# Pydantic Models for Request Bodies
# These classes define the expected JSON payload structure for API endpoints.
# -----------------------------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str
    doc_id: Optional[Union[str, List[str]]] = None
    previous_conversations: Optional[str] = None


class DeleteDocumentRequest(BaseModel):
    doc_id: Optional[str] = None
    document_name: Optional[str] = None


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

class ReduceOutput(BaseModel):
    summary: str
    key_points: list  
    detailed_explanation: str
    
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def extract_rating(text: str) -> int:
    match = re.search(r'\(Rating:\s*(\d+)\)', text)
    return int(match.group(1)) if match else 0
            
def clean_empty_chunks():
    """
    Delete Chunk nodes in Neo4j with empty or null 'text' property.
    """
    with driver.session() as session:
        # Delete any Chunk nodes where the 'text' property is NULL or only whitespace.
        query = "MATCH (c:Chunk) WHERE c.text IS NULL OR trim(c.text) = '' DETACH DELETE c"
        session.run(query)
        logger.info("Cleaned empty Chunk nodes.")


def clean_empty_nodes():
    """
    Delete Entity nodes in Neo4j with empty or null 'name' property.
    """
    with driver.session() as session:
        # Delete any Entity nodes where the 'name' property is NULL or only whitespace.
        query = "MATCH (n:Entity) WHERE n.name IS NULL OR trim(n.name) = '' DETACH DELETE n"
        session.run(query)
        logger.info("Cleaned empty Entity nodes.")

async def rewrite_query_if_needed(question: str, conversation_history: Optional[str]) -> str:
    """
    Rewrite a user query into a standalone query if conversation history is provided.
    :param question: The original user query.
    :param conversation_history: The previous conversation context.
    :return: A standalone query string.
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

async def extract_entity_keywords(question: str) -> list:
    """
    Extract entity names from the given question using an LLM.
    :param question: The user question.
    :return: A list of extracted entity keywords.
    """
    prompt = f"Extract entity names from the following question as a JSON list of strings:\n{question}"
    try:
        response = await async_llm.invoke_json([
            {"role": "system", "content": "You are an expert in entity extraction."},
            {"role": "user", "content": prompt}
        ])
        if isinstance(response, list):
            return response
        elif isinstance(response, dict) and "entities" in response:
            return response["entities"]
        else:
            return []
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        # Fallback heuristic: return first two words if extraction fails.
        return question.split()[:2]

def parse_strict_json(response_text: str) -> Union[dict, list]:
    """
    Parse a string into JSON using standard json and fallback to json5 if needed.
    :param response_text: The string to parse.
    :return: The parsed JSON object.
    """
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.warning("json.loads failed, attempting json5 parser: " + str(e))
        try:
            return json5.loads(response_text)
        except Exception as e2:
            logger.error("json5 parsing also failed: " + str(e2))
            raise e2

def clean_text(text: str) -> str:
    """
    Clean text by removing non-printable characters and extra whitespace.
    :param text: The original text.
    :return: The cleaned text.
    """
    if not isinstance(text, str):
        logger.error(f"Received non-string content for text cleaning: {type(text)}")
        return ""
    cleaned_text = ''.join(filter(lambda x: x in string.printable, text.strip()))
    logger.debug(f"Cleaned text: {cleaned_text[:100]}...")
    return cleaned_text

def chunk_text(text: str, max_chunk_size: int = 512) -> List[str]:
    """
    Split text into chunks with a maximum size, preserving sentence boundaries.
    :param text: The input text.
    :param max_chunk_size: The maximum allowed chunk size.
    :return: A list of text chunks.
    """
    sentences = [s.strip() for s in blingfire.text_to_sentences(text).splitlines() if s.strip()]
    unique_sentences = []
    seen = set()
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)
    chunks = []
    current_chunk = ""
    for s in unique_sentences:
        if current_chunk and (len(current_chunk) + len(s) + 1) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = s
        else:
            current_chunk = s if not current_chunk else current_chunk + " " + s
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def run_cypher_query(query: str, parameters: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
    """
    Execute a Cypher query against the Neo4j database.
    :param query: The Cypher query string.
    :param parameters: Optional dictionary of query parameters.
    :return: A list of dictionaries representing query results.
    """
    # Log parameters only if provided.
    if parameters:
        logger.debug(f"Running Cypher query: {query} with parameters: {parameters}")
    else:
        logger.debug(f"Running Cypher query: {query}")
    try:
        with driver.session() as session:
            result = session.run(query, **parameters)
            data = [record.data() for record in result]
            # For logging, filter out any keys related to embeddings.
            data_for_log = []
            for record in data:
                filtered_record = {k: v for k, v in record.items() if k.lower() != "embedding"}
                data_for_log.append(filtered_record)
            logger.debug(f"Query result: {data_for_log}")
            return data
    except Exception as e:
        logger.error(f"Error executing query: {query}. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Neo4j query execution error: {e}")

def format_natural_response(response: str) -> str:
    """
    Format the natural language response from the LLM, parsing JSON if necessary.
    :param response: The raw response string.
    :return: Formatted response string.
    """
    logger.info(f"Raw LLM Response: {repr(response)}")
    response = response.strip()
    if (response.startswith("{") and response.endswith("}")) or (response.startswith("[") and response.endswith("]")):
        try:
            parsed_data = parse_strict_json(response)
            if isinstance(parsed_data, dict) and "summary" in parsed_data:
                summary = parsed_data.get("summary", "No summary available.")
                key_points = parsed_data.get("key_points", [])
                answer = f"{summary}\n\nKey points of note include:\n"
                for idx, point in enumerate(key_points, 1):
                    answer += f"{idx}. {point['point']} (Importance: {point['rating']})\n"
                return answer.strip()
        except Exception as e:
            logger.error("Error parsing JSON: " + str(e))
            return response
    return response

async def resolve_coreferences_in_parts(text: str) -> str:
    """
    Resolve ambiguous pronouns and vague references in text using an LLM.
    :param text: The input text.
    :return: The text with resolved coreferences.
    """
    async_client = AsyncOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL,
                               base_url=OPENAI_BASE_URL, temperature=OPENAI_TEMPERATURE, stop=OPENAI_STOP,
                               timeout=OPENAI_API_TIMEOUT)
    words = text.split()
    parts = [" ".join(words[i:i + COREF_WORD_LIMIT]) for i in range(0, len(words), COREF_WORD_LIMIT)]

    async def process_part(part: str, idx: int) -> str:
        """
        Process a part of the text to resolve coreferences.
        :param part: The text part.
        :param idx: The index of the part.
        :return: The processed text part.
        """
        prompt = (
                "Resolve all ambiguous pronouns and vague references in the following text by replacing them with their appropriate entities. "
                "Ensure no unresolved pronouns like 'he', 'she', 'himself', etc. remain. "
                "If the referenced entity is unclear, make an intelligent guess based on context.\n\n" + part)
        try:
            resolved_text = await async_llm.invoke([
                {"role": "system", "content": "You are a professional text refiner skilled in coreference resolution."},
                {"role": "user", "content": prompt}
            ])
            logger.info(f"Coreference resolution successful for part {idx + 1}/{len(parts)}.")
            return resolved_text.strip()
        except Exception as e:
            logger.warning(f"Coreference resolution failed for part {idx + 1}/{len(parts)}: {e}")
            return part

    resolved_parts = await asyncio.gather(*[process_part(part, idx) for idx, part in enumerate(parts)])
    combined_text = " ".join(resolved_parts)
    filtered_text = re.sub(r'\b(he|she|it|they|him|her|them|his|her|their|its|himself|herself|his partner)\b', '',
                           combined_text, flags=re.IGNORECASE)
    return filtered_text.strip()

def get_refined_system_message() -> str:
    """
    Get a refined system message instructing the LLM to return valid JSON.
    :return: The system message string.
    """
    return "You are a professional assistant. Please provide a detailed answer."

# =============================================================================
# ASYNCHRONOUS OPENAI & EMBEDDING API CLIENT CLASSES
# =============================================================================
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


class AsyncEmbeddingAPIClient:
    """
    Asynchronous client for fetching embeddings via the embedding API.
    """
    def __init__(self):
        self.embedding_api_url = os.getenv("EMBEDDING_API_URL", "http://localhost/api/embed")
        self.timeout = OPENAI_API_TIMEOUT
        self.logger = logging.getLogger("EmbeddingAPIClient")
        self.cache = {}

    def _get_text_hash(self, text: str) -> str:
        """
        Compute an MD5 hash for the input text.
        :param text: The input text.
        :return: The MD5 hash string.
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    async def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding for a given text, using caching.
        :param text: The input text.
        :return: A list of floats representing the embedding.
        """
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
        """
        Get embeddings for a list of texts.
        :param texts: A list of text strings.
        :return: A list of embedding vectors.
        """
        tasks = [self.get_embedding(text) for text in texts]
        return await asyncio.gather(*tasks)


# Global asynchronous client instances for LLM and Embedding API
async_llm = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    base_url=OPENAI_BASE_URL,
    temperature=OPENAI_TEMPERATURE,
    stop=OPENAI_STOP,
    timeout=OPENAI_API_TIMEOUT
)


async_embedding_client = AsyncEmbeddingAPIClient()


# =============================================================================
# GRAPH DATABASE HELPER FUNCTIONS & CLASSES (Asynchronous LLM Calls)
# =============================================================================
class GraphManager:
    """
    Class for constructing and managing the graph data in Neo4j.
    """
    async def build_graph(self, chunks: List[str], metadata_list: List[Dict[str, Any]]) -> None:
        """
        Build the graph in Neo4j by creating Chunk and Entity nodes and their relationships.
        :param chunks: List of text chunks from documents.
        :param metadata_list: List of metadata dictionaries corresponding to each chunk.
        """
        with driver.session() as session:
            for i, (chunk, meta) in enumerate(zip(chunks, metadata_list)):
                if not chunk.strip() or chunk.strip().isdigit():
                    logger.warning(f"Skipping empty or numeric chunk with ID {i}")
                    continue
                try:
                    chunk_embedding = await async_embedding_client.get_embedding(chunk)
                except Exception as e:
                    logger.error(f"Error generating embedding for chunk: {chunk} - {e}")
                    continue
                query = """
                MERGE (c:Chunk {id: $cid, doc_id: $doc_id})
                ON CREATE SET c.text = $text, 
                            c.document_name = $document_name, 
                            c.timestamp = $timestamp,
                            c.embedding = $embedding
                ON MATCH SET c.text = $text, 
                            c.document_name = $document_name, 
                            c.timestamp = $timestamp,
                            c.embedding = $embedding
                """
                session.run(query, cid=i, doc_id=meta["doc_id"], text=chunk,
                            document_name=meta.get("document_name"),
                            timestamp=meta.get("timestamp"), embedding=chunk_embedding)
                logger.info(f"Added chunk {i} with text: {chunk[:100]}...")
                try:
                    summary = await async_llm.invoke([
                        {"role": "system",
                         "content": "Summarize the following text into relationships in the format: Entity1 -> Entity2 [strength: X.X]. Do not add extra commentary."},
                        {"role": "user", "content": chunk}
                    ])
                except Exception as e:
                    logger.error(f"Summarization error on chunk {i}: {e}")
                    summary = ""
                for line in summary.split("\n"):
                    if "->" in line:
                        parts = line.split("->")
                        if len(parts) >= 2:
                            source = parts[0].strip().lower()
                            target = parts[-1].split("[")[0].strip().lower()
                            weight = 1.0
                            match = re.search(r"\[strength:\s*(\d\.\d)\]", line)
                            if match:
                                weight = float(match.group(1))
                            try:
                                # Compute embeddings for the entity names with error handling.
                                source_embedding = await async_embedding_client.get_embedding(source)
                                target_embedding = await async_embedding_client.get_embedding(target)
                            except Exception as e:
                                logger.error(f"Error generating embedding for entity: {e}")
                                continue  # Skip this relationship if embeddings cannot be computed

                            # Updated query to merge Entity nodes with their embeddings
                            rel_query = """
                                MERGE (a:Entity {name: $source, doc_id: $doc_id})
                                ON CREATE SET a.embedding = $source_embedding
                                ON MATCH SET a.embedding = $source_embedding
                                MERGE (b:Entity {name: $target, doc_id: $doc_id})
                                ON CREATE SET b.embedding = $target_embedding
                                ON MATCH SET b.embedding = $target_embedding
                                MERGE (a)-[r:RELATES_TO {weight: $weight}]->(b)
                                WITH a, b
                                MATCH (c:Chunk {id: $cid, doc_id: $doc_id})
                                MERGE (a)-[:MENTIONED_IN]->(c)
                                MERGE (b)-[:MENTIONED_IN]->(c)
                            """
                            session.run(
                                rel_query,
                                source=source,
                                target=target,
                                weight=weight,
                                doc_id=meta["doc_id"],
                                cid=i,
                                source_embedding=source_embedding,
                                target_embedding=target_embedding
                            )
        logger.info("Graph construction complete.")

    async def reproject_graph(self, graph_name: str = GRAPH_NAME, doc_id: Optional[str] = None) -> None:
        """
        Reproject the graph using Neo4j's Graph Data Science (GDS) library.
        :param graph_name: Name of the projected graph.
        :param doc_id: Optional document ID to filter nodes.
        """
        with driver.session() as session:
            exists_record = session.run("CALL gds.graph.exists($graph_name) YIELD exists",
                                        graph_name=graph_name).single()
            if exists_record and exists_record["exists"]:
                session.run("CALL gds.graph.drop($graph_name)", graph_name=graph_name)
            if doc_id:
                session.run("MATCH (n:Entity {doc_id: $doc_id}) SET n:DocEntity", doc_id=doc_id)
                config = {"RELATES_TO": {"orientation": "UNDIRECTED"}}
                session.run("CALL gds.graph.project($graph_name, ['DocEntity'], $config) YIELD graphName",
                            graph_name=graph_name, config=config)
                session.run("MATCH (n:DocEntity {doc_id: $doc_id}) REMOVE n:DocEntity", doc_id=doc_id)
            else:
                config = {"RELATES_TO": {"orientation": "UNDIRECTED"}}
                session.run("CALL gds.graph.project($graph_name, ['Entity'], $config)", graph_name=GRAPH_NAME,
                            config=config)
            logger.info("Graph projection complete.")

# Instance of GraphManager for use in the API.
graph_manager = GraphManager()

class GraphManagerExtended:
    """
    Extended graph manager for additional operations such as storing community summaries.
    """
    def __init__(self, driver):
        self.driver = driver

    async def store_community_summaries(self, doc_id: str) -> None:
        """
        Detect communities in the graph and store their summaries.
        :param doc_id: The document ID to process.
        """
        with self.driver.session() as session:
            communities = detect_communities(doc_id)
            logger.debug(f"Detected communities for doc_id {doc_id}: {communities}")
            llm = AsyncOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, base_url=OPENAI_BASE_URL,
                              temperature=OPENAI_TEMPERATURE, stop=OPENAI_STOP, timeout=OPENAI_API_TIMEOUT)
            if not communities:
                logger.warning(f"No communities detected for doc_id {doc_id}. Creating default summary.")
                default_summary = "No communities detected."
                default_embedding = await async_embedding_client.get_embedding(default_summary)
                store_query = """
                CREATE (cs:CommunitySummary {doc_id: $doc_id, community: 'default', summary: $summary, embedding: $embedding, timestamp: $timestamp})
                """
                session.run(store_query, doc_id=doc_id, summary=default_summary, embedding=default_embedding,
                            timestamp=datetime.now().isoformat())
            else:
                for comm, entities in communities.items():
                    chunk_query = """
                    MATCH (e:Entity {doc_id: $doc_id})-[:MENTIONED_IN]->(c:Chunk)
                    WHERE toLower(e.name) IN $entity_names AND c.text IS NOT NULL AND c.text <> ""
                    WITH collect(DISTINCT c.text) AS texts
                    RETURN apoc.text.join(texts, "\n") AS aggregatedText
                    """
                    result = session.run(chunk_query, doc_id=doc_id, entity_names=[e.lower() for e in entities])
                    record = result.single()
                    aggregated_text = record["aggregatedText"] if record and record["aggregatedText"] else ""
                    if not aggregated_text.strip():
                        logger.warning(f"No content found for community {comm}. Skipping community summary creation.")
                        continue
                    prompt = ("Summarize the following text into a meaningful summary with key insights. "
                              "If the text is too short or unclear, describe the entities and their possible connections. "
                              "Avoid vague fillers.\n\n" + aggregated_text)
                    logger.info(f"Aggregated text for community {comm}: {aggregated_text}")
                    try:
                        summary = await llm.invoke([{"role": "system",
                                                     "content": "You are a professional summarization assistant. Do not use your prior knowledge and only use knowledge from the provided text."},
                                                    {"role": "user", "content": prompt}])
                    except Exception as e:
                        logger.error(f"Failed to generate summary for community {comm}: {e}")
                        summary = "No summary available due to error."
                    logger.debug(f"Storing summary for community {comm}: {summary}")
                    try:
                        summary_embedding = await async_embedding_client.get_embedding(summary)
                    except Exception as e:
                        logger.error(f"Failed to generate embedding for summary of community {comm}: {e}")
                        summary_embedding = []
                    store_query = """
                    MERGE (cs:CommunitySummary {doc_id: $doc_id, community: $community})
                    SET cs.summary = $summary, cs.embedding = $embedding, cs.timestamp = $timestamp
                    """
                    session.run(store_query, doc_id=doc_id, community=comm, summary=summary,
                                embedding=summary_embedding, timestamp=datetime.now().isoformat())

    async def get_stored_community_summaries(self, doc_id: Optional[str] = None) -> dict:
        """
        Retrieve stored community summaries from the graph.
        :param doc_id: Optional document ID to filter summaries.
        :return: Dictionary of community summaries.
        """
        with self.driver.session() as session:
            if doc_id:
                query = """
                MATCH (cs:CommunitySummary {doc_id: $doc_id})
                RETURN cs.community AS community, cs.summary AS summary, cs.embedding AS embedding
                """
                results = session.run(query, doc_id=doc_id)
            else:
                query = """
                MATCH (cs:CommunitySummary)
                RETURN cs.community AS community, cs.summary AS summary, cs.embedding AS embedding
                """
                results = session.run(query)
            summaries = {}
            for record in results:
                comm = record["community"]
                summaries[comm] = {"summary": record["summary"], "embedding": record["embedding"]}
            return summaries

def detect_communities(doc_id: Optional[str] = None) -> dict:
    """
    Detect communities within the graph using Neo4j's GDS library.
    :param doc_id: Optional document ID to filter nodes.
    :return: A dictionary mapping community IDs to lists of entity names.
    """
    communities = {}
    with driver.session() as session:
        exists_record = session.run("CALL gds.graph.exists($graph_name) YIELD exists", graph_name=GRAPH_NAME).single()
        if exists_record and exists_record["exists"]:
            session.run("CALL gds.graph.drop($graph_name)", graph_name=GRAPH_NAME)
        if doc_id:
            session.run("MATCH (n:Entity {doc_id: $doc_id}) SET n:DocEntity", doc_id=doc_id)
            config = {"RELATES_TO": {"orientation": "UNDIRECTED"}}
            session.run("CALL gds.graph.project($graph_name, ['DocEntity'], $config) YIELD graphName",
                        graph_name=GRAPH_NAME, config=config)
        else:
            config = {"RELATES_TO": {"orientation": "UNDIRECTED"}}
            session.run("CALL gds.graph.project($graph_name, ['Entity'], $config)", graph_name=GRAPH_NAME,
                        config=config)
        result = session.run("CALL gds.leiden.stream($graph_name) YIELD nodeId, communityId "
                             "RETURN gds.util.asNode(nodeId).name AS entity, communityId AS community ORDER BY community, entity",
                             graph_name=GRAPH_NAME)
        for record in result:
            comm = record["community"]
            entity = record["entity"]
            communities.setdefault(comm, []).append(entity)
        session.run("CALL gds.graph.drop($graph_name)", graph_name=GRAPH_NAME)
        if doc_id:
            session.run("MATCH (n:DocEntity {doc_id: $doc_id}) REMOVE n:DocEntity", doc_id=doc_id)
    return communities

class GraphManagerWrapper:
    """
    Wrapper class to abstract graph management operations.
    """
    def __init__(self):
        self.manager = graph_manager

    async def build_graph(self, chunks: List[str], metadata_list: List[Dict[str, Any]]) -> None:
        """
        Build the graph using provided text chunks and metadata.
        """
        await self.manager.build_graph(chunks, metadata_list)

    async def reproject_graph(self, graph_name: str = GRAPH_NAME, doc_id: Optional[str] = None) -> None:
        """
        Reproject the graph.
        """
        await self.manager.reproject_graph(graph_name, doc_id)

    async def store_community_summaries(self, doc_id: str) -> None:
        """
        Store community summaries for the given document ID.
        """
        extended = GraphManagerExtended(driver)
        await extended.store_community_summaries(doc_id)

    async def get_stored_community_summaries(self, doc_id: Optional[str] = None) -> dict:
        """
        Retrieve stored community summaries.
        """
        extended = GraphManagerExtended(driver)
        return await extended.get_stored_community_summaries(doc_id)

# Instantiate the GraphManagerWrapper for API usage.
graph_manager_wrapper = GraphManagerWrapper()
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
       "similarity_threshold": 0.4,  # adjust as needed
       "limit": 30
    }
    similar_chunks = await asyncio.to_thread(run_cypher_query, cypher, params)

    # Step 3: From the similar chunks, collect related document IDs and fetch community summaries.
    related_doc_ids = list({record["doc_id"] for record in similar_chunks})
    community_summaries = {}
    for doc in related_doc_ids:
         query_cs = """
         MATCH (cs:CommunitySummary {doc_id: $doc_id})
         RETURN cs.community AS community, cs.summary AS summary, cs.embedding AS embedding
         """
         cs_result = await asyncio.to_thread(run_cypher_query, query_cs, {"doc_id": doc})
         for record in cs_result:
             community_summaries[record["community"]] = {
                 "summary": record["summary"],
                 "embedding": record["embedding"]
             }
    if not community_summaries:
         raise HTTPException(status_code=500, detail="No related community summaries found based on chunk similarity.")

    # Step 4: Process community summaries to extract key points using plain text.
    # Instead of expecting JSON, we instruct the LLM to list key points as lines in plain text.
    community_reports = list(community_summaries.values())
    random.shuffle(community_reports)
    # (Optionally, if you wish to filter or score these summaries, you can integrate that logic here.)
    intermediate_points = []
    for report in community_reports:
         # Split each community summary into smaller text chunks.
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
                 batch_prompt += f"Chunk {idx+1}:\n\"\"\"\n{chunk}\n\"\"\"\n\n"
             batch_prompt += f"User Query: \"{question}\"\n"
             try:
                 response = await async_llm.invoke([
                     {"role": "system", "content": "You are a professional extraction assistant."},
                     {"role": "user", "content": batch_prompt}
                 ])
                 # Append the response (plain text key points).
                 intermediate_points.append(response.strip())
             except Exception as e:
                 logger.error(f"Batch processing error: {e}")
    
    # Combine all extracted key points into one aggregated text block.
    aggregated_key_points = "\n".join(intermediate_points)
    
    # Optionally, you can add a debug log for the aggregated key points.
    logger.info("Aggregated Key Points:\n%s", aggregated_key_points)
    
    conv_text = f"Conversation History: {conversation_history}\n" if conversation_history else ""
    
    # Step 5: Build a reduction prompt that uses the aggregated key points
    # to generate a final detailed answer in plain text.
    reduce_prompt = f"""
{conv_text}You are a professional assistant tasked with synthesizing a detailed answer from the following extracted key points:
{aggregated_key_points}

User Query: "{question}"

Using only the above key points, generate a comprehensive and detailed answer in plain text that directly addresses the query. Please include a clear summary and explanation that ties together all key points.
"""
    final_answer = await async_llm.invoke([
         {"role": "system", "content": "You are a professional assistant providing detailed answers."},
         {"role": "user", "content": reduce_prompt}
    ])
    
    return final_answer
    
async def global_search_map_reduce(question: str, conversation_history: str = None,
                                   doc_id: str = None,
                                   chunk_size: int = GLOBAL_SEARCH_CHUNK_SIZE, top_n: int = GLOBAL_SEARCH_TOP_N,
                                   batch_size: int = GLOBAL_SEARCH_BATCH_SIZE) -> str:
    """
    Perform global search by dynamically selecting community summaries, extracting key points,
    and reducing the information to generate a final answer.
    :param question: The user query.
    :param conversation_history: Optional conversation context.
    :param doc_id: Optional document ID.
    :param chunk_size: Maximum chunk size for text segmentation.
    :param top_n: Number of top key points to consider.
    :param batch_size: Batch size for processing chunks.
    :return: A detailed final answer as a string.
    """
    # Step 1: Compute query embedding
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
       "limit": 30
    }
    similar_chunks = await asyncio.to_thread(run_cypher_query, cypher, params)
    
    # Step 3: Retrieve related community summaries based on doc_ids
    related_doc_ids = list({record["doc_id"] for record in similar_chunks})
    community_summaries = {}
    for doc in related_doc_ids:
         query_cs = """
         MATCH (cs:CommunitySummary {doc_id: $doc_id})
         RETURN cs.community AS community, cs.summary AS summary, cs.embedding AS embedding
         """
         cs_result = await asyncio.to_thread(run_cypher_query, query_cs, {"doc_id": doc})
         for record in cs_result:
             community_summaries[record["community"]] = {
                 "summary": record["summary"],
                 "embedding": record["embedding"]
             }
    
    if not community_summaries:
         raise HTTPException(status_code=500, detail="No related community summaries found based on chunk similarity.")
    
    
    # Step 4: Continue with the remaining logic (chunking, extracting key points, and reducing).
    community_reports = list(community_summaries.values())
    random.shuffle(community_reports)
    selected_reports_with_scores = await asyncio.to_thread(select_relevant_communities, question, community_reports)
    if not selected_reports_with_scores:
        selected_reports_with_scores = [(report, 0) for report in community_reports[:top_n]]
    
    intermediate_points = []
    for report, _ in selected_reports_with_scores:
        # Break the community summary into manageable chunks.
        chunks = chunk_text(report, max_chunk_size=chunk_size)
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_prompt = (
                "You are an expert in extracting key information. For each of the following community report chunks, "
                "extract key points that are relevant to answering the user query provided at the end. "
                "Return the output strictly as a valid JSON array in this format: "
                '[{"point": "key detail", "rating": 1-100}].\n\n'
                "**IMPORTANT:** Use **double quotes** for keys and string values. Do NOT add any explanation, commentary, or extra text outside the JSON structure."
            )
            for idx, chunk in enumerate(batch):
                batch_prompt += f"\n\nChunk {idx}:\n\"\"\"\n{chunk}\n\"\"\""
            batch_prompt += f"\n\nUser Query:\n\"\"\"\n{question}\n\"\"\""
            try:
                response = await async_llm.invoke_json(
                    [{"role": "system", "content": "You are a professional extraction assistant."},
                     {"role": "user", "content": batch_prompt}]
                )
                points = response
                intermediate_points.extend(points)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
    
    # Sort the extracted key points by their rating.
    intermediate_points_sorted = sorted(
        intermediate_points,
        key=lambda x: extract_rating(x),
        reverse=True
    )

    selected_points = intermediate_points_sorted[:top_n]
    aggregated_context = "\n".join(
        [f"{pt['point']} (Rating: {pt['rating']})" if isinstance(pt, dict) else f"{pt} (Rating: 0)" for pt in selected_points]
    )
    
    conv_text = f"Conversation History: {conversation_history}\n" if conversation_history else ""
    
    # Build the final reduce prompt.
    reduce_prompt = f"""
    {conv_text}You are a professional assistant specialized in synthesizing detailed information from provided data.
    Your task is to analyze the following list of intermediate key points and the original user query, and then generate a comprehensive answer that fully explains the topic using only the provided information.
    Using ONLY the following intermediate key points:
    {json.dumps(intermediate_points_sorted[:top_n], indent=2)}
    and the original user query: "{question}",
    generate a detailed answer that directly addresses the query.
    Please follow these instructions:
    1. Produce a detailed summary that directly and thoroughly answers the user’s query.
    2. Include a list of the key points along with their ratings exactly as provided.
    3. Provide an extended detailed explanation that weaves together all key points, elaborating on how each contributes to the overall answer.
    4. Do not use any external knowledge; rely solely on the provided key points.
    5. Return your response strictly in the following JSON format:
        {{
        "summary": "Your detailed answer summary here",
        "key_points": [
            {{"point": "key detail", "rating": "1-100"}},
            {{"point": "another detail", "rating": "1-100"}}
        ],
        "detailed_explanation": "A comprehensive explanation that connects all key points and fully addresses the query."
        }}
    Ensure that your answer is extensive, uses complete sentences, and provides a deep, detailed explanation based solely on the supplied context.
    """
    
    try:
        final_answer = await async_llm.invoke_json([
            {"role": "system", "content": "You are a professional assistant providing detailed answers based solely on provided information. Infer insights when possible. Your response MUST be a valid JSON object."},
            {"role": "user", "content": reduce_prompt}
        ])
        if isinstance(final_answer, dict) and "summary" in final_answer:
            return format_natural_response(json.dumps(final_answer))
        return "Here's the provided content:\n\n" + json.dumps(final_answer, indent=2)
    except Exception as e:
        final_answer = f"Error during reduce step: {e}"
    return final_answer

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute the cosine similarity between two vectors.
    :param vec1: First vector.
    :param vec2: Second vector.
    :return: Cosine similarity as a float.
    """
    from numpy import dot
    from numpy.linalg import norm
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2) + 1e-8)

def select_relevant_communities(query: str, community_reports: List[Union[str, dict]], top_k: int = GLOBAL_SEARCH_TOP_N,
                                threshold: float = RELEVANCE_THRESHOLD) -> List[Tuple[str, float]]:
    """
    Select and rank community reports that are relevant to the query using cosine similarity.
    :param query: The user query.
    :param community_reports: List of community reports (either as strings or dicts with embeddings).
    :param top_k: Number of top reports to select.
    :param threshold: Relevance threshold for filtering.
    :return: List of tuples with report snippet and similarity score.
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

# =============================================================================
# FASTAPI ENDPOINTS
# =============================================================================
app = FastAPI(title="Graph RAG API", description="End-to-end Graph Database RAG on Neo4j", version="1.0.0")

@app.post("/upload_documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload documents, process them into text chunks, build the graph, and store community summaries.
    :param files: List of files uploaded.
    :return: JSON message confirming successful processing.
    """
    document_texts = []
    metadata = []
    for file in files:
        filename = file.filename.lower()
        try:
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
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}")
        text = clean_text(text)
        text = await resolve_coreferences_in_parts(text)
        doc_id = str(uuid.uuid4())
        metadata.append({"doc_id": doc_id, "document_name": file.filename, "timestamp": datetime.now().isoformat()})
        document_texts.append(text)
    all_chunks = []
    meta_list = []
    for text, meta in zip(document_texts, metadata):
        chunks = chunk_text(text, max_chunk_size=CHUNK_SIZE_GDS)
        logger.info(f"Generated {len(chunks)} chunks for document {meta['document_name']}")
        all_chunks.extend(chunks)
        meta_list.extend([meta] * len(chunks))
    try:
        await graph_manager_wrapper.build_graph(all_chunks, meta_list)
    except Exception as e:
        logger.error(f"Graph building error: {e}")
        raise HTTPException(status_code=500, detail=f"Graph building error: {e}")
    clean_empty_chunks()
    clean_empty_nodes()
    unique_doc_ids = set(meta["doc_id"] for meta in metadata)
    for doc_id in unique_doc_ids:
        try:
            await graph_manager_wrapper.store_community_summaries(doc_id)
        except Exception as e:
            logger.error(f"Error storing community summaries for doc_id {doc_id}: {e}")
    return {"message": "Documents processed, graph updated, and community summaries stored successfully."}

@app.post("/cypher_search")
async def cypher_search(request: QuestionRequest):
    """
    Execute a cypher search based on the user question by matching entities and retrieving related text chunks.
    :param request: The QuestionRequest object containing query details.
    :return: JSON containing the final answer, aggregated text, and executed queries.
    """
    # Step 1: Extract candidate entities from the user query.
    prompt = f"Extract entity names from the following question. Provide the names separated by commas:\n{request.question}"
    try:
        response = await async_llm.invoke([
            {"role": "system", "content": "You are an expert in entity extraction."},
            {"role": "user", "content": prompt}
        ])
        candidate_entities = [e.strip() for e in response.split(",") if e.strip()]
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        candidate_entities = request.question.split()[:2]
    final_entities = []

    # For each candidate, compute similarity directly in Neo4j.
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
                # Execute the similarity query in a separate thread.
                result = await asyncio.to_thread(run_cypher_query, cypher_query, {"candidate_embedding": candidate_embedding})
            except Exception as e:
                logger.error(f"Error executing similarity query for candidate '{candidate}': {e}")
                continue

            if result and len(result) > 0:
                top_match = result[0].get("name")
                if top_match:
                    final_entities.append(top_match)

    # Remove duplicate entity matches.
    final_entities = list(set(final_entities))

    results = []
    # Step 2: If no matching entities were found, use the fallback query.
    if not final_entities:
        fallback_query = f"""
        MATCH (e:Entity)
        WHERE apoc.text.jaroWinklerDistance(e.name, "{request.question}") > 0.8
        WITH e
        MATCH (e)-[:RELATES_TO]->(related:Entity)
        OPTIONAL MATCH (related)-[:MENTIONED_IN]->(chunk:Chunk)
        RETURN e.name AS MatchedEntity, related.name AS RelatedEntity, collect(chunk.text) AS ChunkTexts
        LIMIT {CYPHER_QUERY_LIMIT};
        """
        try:
            query_result = await asyncio.to_thread(run_cypher_query, fallback_query)
            results.append(query_result)
        except Exception as e:
            logger.error(f"Error executing fallback query: {e}")
            raise HTTPException(status_code=500, detail=f"Error executing fallback query: {e}")
    else:
        # Step 3: For each extracted entity, generate and execute three Cypher queries.
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
            # Execute each query and append results.
            for q in [query_a, query_b, query_c]:
                try:
                    qr = await asyncio.to_thread(run_cypher_query, q)
                    results.append(qr)
                except Exception as e:
                    logger.error(f"Error executing query for entity '{entity}': {e}")

    # Step 4: Map Reduce – aggregate all returned text chunks without duplicates.
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

    # Step 5: Use the aggregated text (plus the original query) to generate a final answer.
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

# -----------------------------------------------------------------------------
# Endpoint: Global Search
# -----------------------------------------------------------------------------
@app.post("/global_search")
async def global_search(request: GlobalSearchRequest):
    # Rewrite query if previous conversation exists.
    request.question = await rewrite_query_if_needed(request.question, request.previous_conversations)
    final_answer_text = await global_search_map_reduce_plain(
         question=request.question,
         conversation_history=request.previous_conversations,
         doc_id=request.doc_id,
         chunk_size=GLOBAL_SEARCH_CHUNK_SIZE,
         top_n=GLOBAL_SEARCH_TOP_N,
         batch_size=GLOBAL_SEARCH_BATCH_SIZE
    )
    # Return the plain text answer wrapped in JSON.
    return {"answer": final_answer_text}

# -----------------------------------------------------------------------------
# Endpoint: Local Search
# -----------------------------------------------------------------------------
@app.post("/local_search")
async def local_search(request: LocalSearchRequest):
    """
    Local search endpoint that uses conversation history and document context to generate an answer.
    :param request: LocalSearchRequest object.
    :return: JSON with the local search answer.
    """
    # Rewrite the query if previous conversation exists.
    request.question = await rewrite_query_if_needed(request.question, request.previous_conversations)

    conversation_context = (
        f"Conversation History:\n{request.previous_conversations}\n\n"
        if request.previous_conversations else ""
    )

    if request.doc_id:
        summaries = await graph_manager_wrapper.get_stored_community_summaries(doc_id=request.doc_id)
    else:
        summaries = await graph_manager_wrapper.get_stored_community_summaries()

    if summaries:
        query_embedding = await async_embedding_client.get_embedding(request.question)
        scored_summaries = []
        for comm, info in summaries.items():
            if info.get("embedding"):
                summary_embedding = info["embedding"]
                score = cosine_similarity(query_embedding, summary_embedding)
                scored_summaries.append((comm, info["summary"], score))
        threshold = 0.3
        filtered_summaries = [
            (comm, summary, score) for comm, summary, score in scored_summaries if score >= threshold
        ]
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
        text_unit_results = await asyncio.to_thread(run_cypher_query, text_unit_query, {"doc_id": request.doc_id})
    else:
        text_unit_query = """
        MATCH (c:Chunk)
        RETURN c.text AS chunk_text
        LIMIT 5
        """
        text_unit_results = await asyncio.to_thread(run_cypher_query, text_unit_query)

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
        {"role": "system",
         "content": "You are a professional assistant who answers strictly based on the provided context."},
        {"role": "user", "content": prompt}
    ])
    return {"local_search_answer": answer}

# -----------------------------------------------------------------------------
# Endpoint: Drift Search
# -----------------------------------------------------------------------------
@app.post("/drift_search")
async def drift_search(request: DriftSearchRequest):
    # Step 1: Rewrite the query if previous conversation exists.
    request.question = await rewrite_query_if_needed(request.question, request.previous_conversations)
    
    # -----------------------------
    # Global Search: Cosine Similarity Filtering
    # -----------------------------
    # Compute the query embedding.
    query_embedding = await async_embedding_client.get_embedding(request.question)
    
    # Run a Cypher query to retrieve Chunk nodes similar to the query embedding.
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
       "similarity_threshold": 0.4,  # Adjust threshold as needed.
       "limit": 30
    }
    similar_chunks = await asyncio.to_thread(run_cypher_query, cypher, params)
    
    # Extract unique doc_ids from the similar chunks.
    related_doc_ids = list({record["doc_id"] for record in similar_chunks})
    
    # Retrieve community summaries for these document IDs.
    community_summaries = {}
    for doc in related_doc_ids:
         query_cs = """
         MATCH (cs:CommunitySummary {doc_id: $doc_id})
         RETURN cs.community AS community, cs.summary AS summary, cs.embedding AS embedding
         """
         cs_result = await asyncio.to_thread(run_cypher_query, query_cs, {"doc_id": doc})
         for record in cs_result:
             community_summaries[record["community"]] = {
                 "summary": record["summary"],
                 "embedding": record["embedding"]
             }
    if not community_summaries:
         raise HTTPException(status_code=500, detail="No related community summaries found based on chunk similarity.")
    
    # Build a global context by concatenating all retrieved community summaries.
    community_reports = [v["summary"] for v in community_summaries.values() if v.get("summary")]
    global_context = "\n\n".join(community_reports)
    
    conversation_context = f"Conversation History:\n{request.previous_conversations}\n\n" if request.previous_conversations else ""
    
    # -----------------------------
    # Primer Phase: Generate Intermediate Answer & Follow-Up Questions
    # -----------------------------
    primer_prompt = f"""
You are an expert in synthesizing information from diverse community reports.
Based on the following global context derived from document similarity filtering:
{global_context}
{conversation_context}
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
    
    # Parse the primer result into an intermediate answer and follow-up questions.
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
                         line = line[dot_index+1:].strip()
                 follow_up_questions.append(line)
    else:
         # Fallback: treat the entire output as the intermediate answer.
         intermediate_answer = primer_result.strip()
    
    drift_hierarchy = {
         "query": request.question,
         "answer": intermediate_answer,
         "follow_ups": []
    }
    
    # -----------------------------
    # Local Phase: Process Each Follow-Up Question Using Local Document Context
    # -----------------------------
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
         chunk_results = await asyncio.to_thread(run_cypher_query, local_chunk_query, params)
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
                             line = line[dot_index+1:].strip()
                     local_follow_ups.append(line)
         else:
             local_answer = local_result.strip()
         
         drift_hierarchy["follow_ups"].append({
             "query": follow_up,
             "answer": local_answer,
             "follow_ups": local_follow_ups
         })
    
    # -----------------------------
    # Reduction Phase: Synthesize Final Answer Using the Full Hierarchy
    # -----------------------------
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


# -----------------------------------------------------------------------------
# Endpoint: List Documents
# -----------------------------------------------------------------------------
@app.get("/documents")
async def list_documents():
    """
    List all distinct documents based on chunks in the graph.
    :return: JSON with a list of documents.
    """
    try:
        query = """
            MATCH (c:Chunk)
            RETURN DISTINCT c.doc_id AS doc_id, c.document_name AS document_name, c.timestamp AS timestamp
        """
        results = await asyncio.to_thread(run_cypher_query, query)
        return {"documents": results}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Error listing documents")

# -----------------------------------------------------------------------------
# Endpoint: Delete Document
# -----------------------------------------------------------------------------
@app.delete("/delete_document")
async def delete_document(request: DeleteDocumentRequest):
    """
    Delete a document from the graph based on doc_id or document_name.
    :param request: DeleteDocumentRequest object.
    :return: JSON message confirming deletion.
    """
    if not request.doc_id and not request.document_name:
        raise HTTPException(status_code=400, detail="Provide either doc_id or document_name to delete a document.")
    try:
        if request.doc_id:
            query = "MATCH (n) WHERE n.doc_id = $doc_id DETACH DELETE n"
            await asyncio.to_thread(run_cypher_query, query, {"doc_id": request.doc_id})
            message = f"Document with doc_id {request.doc_id} deleted successfully."
        else:
            query = "MATCH (n) WHERE n.document_name = $document_name DETACH DELETE n"
            await asyncio.to_thread(run_cypher_query, query, {"document_name": request.document_name})
            message = f"Document with name {request.document_name} deleted successfully."
        return {"message": message}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {e}")

# -----------------------------------------------------------------------------
# Endpoint: Get Communities
# -----------------------------------------------------------------------------
@app.get("/communities")
async def get_communities(doc_id: Optional[str] = None):
    """
    Retrieve detected communities from the graph.
    :param doc_id: Optional document ID filter.
    :return: JSON with communities.
    """
    try:
        communities = detect_communities(doc_id)
        return {"communities": communities}
    except Exception as e:
        logger.error(f"Error detecting communities: {e}")
        raise HTTPException(status_code=500, detail=f"Error detecting communities: {e}")

# -----------------------------------------------------------------------------
# Endpoint: Get Community Summaries
# -----------------------------------------------------------------------------
@app.get("/community_summaries")
async def community_summaries(doc_id: Optional[str] = None):
    """
    Retrieve stored community summaries.
    :param doc_id: Optional document ID filter.
    :return: JSON with community summaries.
    """
    try:
        summaries = await graph_manager_wrapper.get_stored_community_summaries(doc_id)
        return {"community_summaries": summaries}
    except Exception as e:
        logger.error(f"Error generating community summaries: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating community summaries: {e}")

# -----------------------------------------------------------------------------
# Root Endpoint
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    """
    Root endpoint for basic health check.
    :return: JSON message indicating that the API is running.
    """
    return {"message": "GraphRAG API is running."}

# -----------------------------------------------------------------------------
# Main entry point: Start the application using Uvicorn.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
