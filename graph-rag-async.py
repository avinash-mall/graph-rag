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
# Helper function to run asynchronous coroutines without calling asyncio.run()
# inside an already running event loop.
# -----------------------------------------------------------------------------
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        result_container = {}
        def run():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result_container["result"] = new_loop.run_until_complete(coro)
            except Exception as e:
                result_container["result"] = e
            finally:
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
load_dotenv()

# -----------------------------------------------------------------------------
# Application & Global Configuration
# -----------------------------------------------------------------------------
APP_HOST = os.getenv("APP_HOST") or "0.0.0.0"
APP_PORT = int(os.getenv("APP_PORT") or "8000")

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
# Neo4j Driver Setup
# -----------------------------------------------------------------------------
driver = GraphDatabase.driver(DB_URL, auth=(DB_USERNAME, DB_PASSWORD))

# -----------------------------------------------------------------------------
# Pydantic Models for Request Bodies
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

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_schema_from_neo4j() -> str:
    try:
        # Run the APOC procedure to fetch the schema
        schema_result = run_cypher_query("CALL apoc.meta.schema()")
        # Optionally, process the result to format it nicely for your LLM
        return json.dumps(schema_result, indent=2)
    except Exception as e:
        logger.error(f"Error retrieving schema: {e}")
        return "Error retrieving schema from Neo4j."
        
async def validate_and_refine_query(initial_query: str, question: str, llm_instance: AsyncOpenAI) -> str:
    results = await asyncio.to_thread(run_cypher_query, initial_query)
    if results:
        return initial_query
    fallback_prompt = (
        f"The generated Cypher query returned no results. "
        f"Based on the provided schema and question, generate an alternative query that retrieves data. "
        f"Schema: (:Actor)-[:ActedIn]->(:Movie). Question: {question}. "
        "Return valid JSON with key 'cypher_query'."
    )
    messages = [
        {"role": "system", "content": "You are a professional assistant. Return ONLY valid JSON with no extra commentary."},
        {"role": "user", "content": fallback_prompt}
    ]
    raw_response = await llm_instance.invoke_json(messages, fallback=True)
    if isinstance(raw_response, (dict, list)):
        raw_response = json.dumps(raw_response)
    refined_query = extract_cypher_query(raw_response)
    fallback_results = await asyncio.to_thread(run_cypher_query, refined_query)
    return refined_query if fallback_results else refined_query

def build_enhanced_final_prompt(question: str, responses: List[str], query: str, query_output: dict) -> str:
    return (
        f"User Question: {question}\n\n"
        f"Generated Cypher Query: {query}\n\n"
        f"Query Output: {json.dumps(query_output, indent=2)}\n\n"
        "Based solely on the provided data, generate a comprehensive answer. "
        "If the query output is empty, explain possible reasons (e.g., schema mismatch or missing data) "
        "and suggest corrective actions. Do not include external knowledge."
    )

def extract_cypher_query(response_text: str) -> str:
    try:
        data = parse_strict_json(response_text)
        if isinstance(data, dict):
            if "cypher_query" in data:
                return data["cypher_query"].strip()
            elif "query" in data:
                return data["query"].strip()
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}")
    return _postprocess_output_cypher(response_text)

def fix_cypher_query(query: str) -> str:
    # If the query is wrapped in a JSON object with key "query", extract it.
    try:
        data = json.loads(query)
        if isinstance(data, dict) and "query" in data:
            query = data["query"]
    except Exception:
        pass

    # Replace invalid label syntax with a valid pattern.
    if "b:Entity|Chunk" in query:
        query = query.replace("b:Entity|Chunk", "b")
        # If no WHERE clause is present in the MATCH clause, add one.
        match_clause = query.split("WITH")[0]
        if "WHERE" not in match_clause:
            query = query.replace("MATCH (a:Entity)-[r]->(b)", "MATCH (a:Entity)-[r]->(b)\nWHERE b:Entity OR b:Chunk")
    # Replace type(b) with type(r) since b is a node.
    if "type(b)" in query:
        query = query.replace("type(b)", "type(r)")
    return query

def _postprocess_output_cypher(output_cypher: str) -> str:
    if "```" in output_cypher:
        parts = output_cypher.split("```")
        code = parts[1]
        if code.lower().startswith("cypher"):
            code = code[len("cypher"):].strip()
        return code.strip()
    partition_by = "**Explanation:**"
    output_cypher, _, _ = output_cypher.partition(partition_by)
    return output_cypher.strip("`\n").lstrip("cypher\n").strip("`\n ")

def parse_strict_json(response_text: str) -> Union[dict, list]:
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
    if not isinstance(text, str):
        logger.error(f"Received non-string content for text cleaning: {type(text)}")
        return ""
    cleaned_text = ''.join(filter(lambda x: x in string.printable, text.strip()))
    logger.debug(f"Cleaned text: {cleaned_text[:100]}...")
    return cleaned_text

def chunk_text(text: str, max_chunk_size: int = 512) -> List[str]:
    sentences = [s.strip() for s in blingfire.text_to_sentences(text).splitlines() if s.strip()]
    unique_sentences = list(set(sentences))
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
    logger.debug(f"Running Cypher query: {query} with parameters: {parameters}")
    try:
        with driver.session() as session:
            result = session.run(query, **parameters)
            data = [record.data() for record in result]
            logger.debug(f"Query result: {data}")
            return data
    except Exception as e:
        logger.error(f"Error executing query: {query}. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Neo4j query execution error: {e}")

def build_llm_answer_prompt(query_context: dict) -> str:
    prompt = f"""Given the data extracted from the graph database:

Question: {query_context.get('question')}

Standard Query Output:
{json.dumps(query_context.get('standard_query_output'), indent=2)}

Fuzzy Query Output:
{json.dumps(query_context.get('fuzzy_query_output'), indent=2)}

General Query Output:
{json.dumps(query_context.get('general_query_output'), indent=2)}

Provide a detailed answer to the query based solely on the information above."""
    return prompt.strip()

def build_combined_answer_prompt(user_question: str, responses: List[str]) -> str:
    prompt = f"User Question: {user_question}\n\nThe following responses were generated for different aspects of your query:\n"
    for idx, resp in enumerate(responses, start=1):
        prompt += f"{idx}. {resp}\n"
    prompt += "\nBased on the above responses, provide a final comprehensive answer that directly addresses the original question. Do not use your prior knowledge and only use knowledge from the provided text."
    return prompt.strip()

def format_natural_response(response: str) -> str:
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

# --- MISSING LOGIC ADDED BELOW ---

async def resolve_coreferences_in_parts(text: str) -> str:
    async_client = AsyncOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL,
                                 base_url=OPENAI_BASE_URL, temperature=OPENAI_TEMPERATURE, stop=OPENAI_STOP, timeout=OPENAI_API_TIMEOUT)
    words = text.split()
    parts = [" ".join(words[i:i + COREF_WORD_LIMIT]) for i in range(0, len(words), COREF_WORD_LIMIT)]

    async def process_part(part: str, idx: int) -> str:
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

def extract_entities_with_llm(text: str) -> List[Dict[str, str]]:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, base_url=OPENAI_BASE_URL,
                         temperature=OPENAI_TEMPERATURE, stop=OPENAI_STOP)
    prompt = ("Extract all named entities from the following text. "
              "For each entity, provide its name and type (e.g., cardinal value, date value, event name, building name, "
              "geo-political entity, language name, law name, money name, person name, organization name, location name, "
              "affiliation, ordinal value, percent value, product name, quantity value, time value, name of work of art). "
              "If uncertain, label it as 'UNKNOWN'. "
              "Return ONLY a valid JSON array (without any additional text or markdown) in the exact format: "
              '[{"name": "entity name", "type": "entity type"}].'
              "\n\n" + text)
    try:
        response = run_async(client.invoke_json([
            {"role": "system", "content": "You are a professional NER extraction assistant."},
            {"role": "user", "content": prompt}
        ]))
        if isinstance(response, dict) and "entities" in response:
            entities = response["entities"]
        elif isinstance(response, list):
            entities = response
        else:
            raise ValueError("Unexpected JSON structure in NER response")
        filtered_entities = [entity for entity in entities if
                             entity.get('name', '').strip() and entity.get('type', '').strip()]
        return filtered_entities
    except Exception as e:
        logger.error(f"NER extraction error: {e}")
        return []

# --- END OF MISSING LOGIC ---

def get_refined_system_message() -> str:
    return (
        "You are a professional assistant. Your response MUST be a valid JSON object with no additional text, "
        "markdown formatting, or commentary. Ensure the JSON is complete and properly formatted with matching opening "
        "and closing braces. If there is an error, return a JSON object with a key 'error' and a descriptive message."
    )

# =============================================================================
# ASYNCHRONOUS OPENAI & EMBEDDING API CLIENT CLASSES
# =============================================================================
class AsyncOpenAI:
    def __init__(self, api_key: str, model: str, base_url: str, temperature: float, stop: List[str], timeout: int):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.stop = stop
        self.timeout = timeout

    async def invoke(self, messages: List[Dict[str, str]]) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature, "stop": self.stop}
        logger.debug(f"LLM Payload Sent: {json.dumps(payload, indent=2)}")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
            response = await client.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return content

    async def invoke_json(self, messages: List[Dict[str, str]], fallback: bool = True) -> Union[dict, list]:
        primary_messages = [{"role": "system", "content": get_refined_system_message()}] + messages
        response_text = await self.invoke(primary_messages)
        logger.debug("Raw LLM response: " + repr(response_text))
        try:
            parsed = parse_strict_json(response_text)
            return parsed
        except Exception as e:
            logger.warning("Initial JSON parsing failed: " + str(e))
            if fallback:
                fallback_messages = [{"role": "system", "content": get_refined_system_message()}] + messages
                fallback_response_text = await self.invoke(fallback_messages)
                logger.debug("Fallback LLM response:\n" + fallback_response_text)
                try:
                    parsed = parse_strict_json(fallback_response_text)
                    return parsed
                except Exception as e2:
                    logger.error("Fallback JSON parsing failed: " + str(e2))
                    return {"error": "Failed to parse JSON response after fallback."}
            else:
                return {"error": "Failed to parse JSON response."}

class AsyncEmbeddingAPIClient:
    def __init__(self):
        self.embedding_api_url = os.getenv("EMBEDDING_API_URL", "http://localhost/api/embed")
        self.timeout = OPENAI_API_TIMEOUT
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
            response = await client.post(self.embedding_api_url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            embedding = response.json().get("embeddings", [[]])[0]
            if not embedding:
                raise ValueError(f"Empty embedding returned for text: {text[:50]}")
            self.logger.debug("Received embedding of length: %d", len(embedding))
            self.cache[text_hash] = embedding
            return embedding

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        tasks = [self.get_embedding(text) for text in texts]
        return await asyncio.gather(*tasks)

# Global asynchronous client instances
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
    async def build_graph(self, chunks: List[str], metadata_list: List[Dict[str, Any]]) -> None:
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
                        {"role": "system", "content": "Summarize the following text into relationships in the format: Entity1 -> Entity2 [strength: X.X]. Do not add extra commentary."},
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
                            rel_query = """
                                MERGE (a:Entity {name: $source, doc_id: $doc_id})
                                MERGE (b:Entity {name: $target, doc_id: $doc_id})
                                MERGE (a)-[r:RELATES_TO {weight: $weight}]->(b)
                                WITH a, b
                                MATCH (c:Chunk {id: $cid, doc_id: $doc_id})
                                MERGE (a)-[:MENTIONED_IN]->(c)
                                MERGE (b)-[:MENTIONED_IN]->(c)
                            """
                            session.run(rel_query, source=source, target=target, weight=weight, doc_id=meta["doc_id"],
                                        cid=i)
        logger.info("Graph construction complete.")

    async def reproject_graph(self, graph_name: str = GRAPH_NAME, doc_id: Optional[str] = None) -> None:
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

graph_manager = GraphManager()

class GraphManagerExtended:
    def __init__(self, driver):
        self.driver = driver

    async def store_community_summaries(self, doc_id: str) -> None:
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
    def __init__(self):
        self.manager = graph_manager

    async def build_graph(self, chunks: List[str], metadata_list: List[Dict[str, Any]]) -> None:
        await self.manager.build_graph(chunks, metadata_list)

    async def reproject_graph(self, graph_name: str = GRAPH_NAME, doc_id: Optional[str] = None) -> None:
        await self.manager.reproject_graph(graph_name, doc_id)

    async def store_community_summaries(self, doc_id: str) -> None:
        extended = GraphManagerExtended(driver)
        await extended.store_community_summaries(doc_id)

    async def get_stored_community_summaries(self, doc_id: Optional[str] = None) -> dict:
        extended = GraphManagerExtended(driver)
        return await extended.get_stored_community_summaries(doc_id)

graph_manager_wrapper = GraphManagerWrapper()

async def global_search_map_reduce(question: str, conversation_history: Optional[str] = None, doc_id: Optional[str] = None,
                             chunk_size: int = GLOBAL_SEARCH_CHUNK_SIZE, top_n: int = GLOBAL_SEARCH_TOP_N,
                             batch_size: int = GLOBAL_SEARCH_BATCH_SIZE) -> str:
    summaries = await graph_manager_wrapper.get_stored_community_summaries(doc_id=doc_id)
    if not summaries:
        raise HTTPException(status_code=500, detail="No community summaries available. Please re-index the dataset.")
    community_reports = list(summaries.values())
    random.shuffle(community_reports)
    selected_reports_with_scores = await asyncio.to_thread(select_relevant_communities, question, community_reports)
    if not selected_reports_with_scores:
        selected_reports_with_scores = [(report, 0) for report in community_reports[:top_n]]
    logger.info("Selected community reports for dynamic selection:")
    for report, score in selected_reports_with_scores:
        logger.info("Score: %.4f, Snippet: %s", score, report[:100])
    intermediate_points = []
    for report, _ in selected_reports_with_scores:
        chunks = chunk_text(report, max_chunk_size=chunk_size)
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_prompt = (
                "You are an expert in extracting key information. For each of the following community report chunks, "
                "extract key points that are relevant to answering the user query provided at the end. "
                "Return the output strictly as a valid JSON array in this format: "
                '[{"point": "key detail", "rating": 1-100}].\n\n'
                "**IMPORTANT:** Use **double quotes** for keys and string values. Do NOT add any explanation, commentary, or extra text outside the JSON structure.")
            for idx, chunk in enumerate(batch):
                batch_prompt += f"\n\nChunk {idx}:\n\"\"\"\n{chunk}\n\"\"\""
            batch_prompt += f"\n\nUser Query:\n\"\"\"\n{question}\n\"\"\""
            try:
                response = await async_llm.invoke_json(
                    [{"role": "system", "content": "You are a professional extraction assistant."},
                     {"role": "user", "content": batch_prompt}])
                points = response
                intermediate_points.extend(points)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
    logger.info("Intermediate key points extracted (unsorted):")
    for pt in intermediate_points:
        logger.info("Key point: %s, Rating: %s", pt.get("point"), pt.get("rating"))
    intermediate_points_sorted = sorted(intermediate_points, key=lambda x: x.get("rating") or 0, reverse=True)
    selected_points = intermediate_points_sorted[:top_n]
    aggregated_context = "\n".join([f"{pt['point']} (Rating: {pt['rating']})" for pt in selected_points])
    conv_text = f"Conversation History: {conversation_history}\n" if conversation_history else ""
    reduce_prompt = f"""
        You are a professional assistant specialized in synthesizing detailed information from provided data.
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
        final_answer = await async_llm.invoke_json([{"role": "system",
                                                  "content": "You are a professional assistant providing detailed answers based solely on provided information. Infer insights when possible. Your response MUST be a valid JSON object."},
                                                 {"role": "user", "content": reduce_prompt}])
        if isinstance(final_answer, dict) and "summary" in final_answer:
            return format_natural_response(json.dumps(final_answer))
        return "Here's the provided content:\n\n" + json.dumps(final_answer, indent=2)
    except Exception as e:
        final_answer = f"Error during reduce step: {e}"
    return final_answer

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    from numpy import dot
    from numpy.linalg import norm
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2) + 1e-8)

def select_relevant_communities(query: str, community_reports: List[Union[str, dict]], top_k: int = GLOBAL_SEARCH_TOP_N,
                                threshold: float = RELEVANCE_THRESHOLD) -> List[Tuple[str, float]]:
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
# TEXT2CYPHER RETRIEVER (Asynchronous Version)
# =============================================================================
class Text2CypherRetriever:
    def __init__(self, driver, llm_instance: AsyncOpenAI, schema: str = "(:Entity)-[:MENTIONED_IN]->(:Chunk)"):
        self.driver = driver
        self.llm = llm_instance
        self.schema = schema

    async def get_cypher(self, question: str) -> List[dict]:
        instruction = (
            "Generate a Cypher query that not only retrieves entities (e.g., names, organization, location) and their relationships but also fetches the 'text' field from all related Chunk nodes. This text data should be used as context when generating the comprehensive answer."
        )
        prompt = instruction.format(schema=self.schema, question=question)
        messages = [
            {"role": "system", "content": "You are a professional assistant. Your response MUST be valid JSON with no extra commentary."},
            {"role": "user", "content": prompt}
        ]
        raw_response = await self.llm.invoke_json(messages, fallback=True)
        if isinstance(raw_response, (dict, list)):
            raw_response = json.dumps(raw_response)
        cypher_query = extract_cypher_query(raw_response)
        cypher_query = fix_cypher_query(cypher_query)
        validated_query = await validate_and_refine_query(cypher_query, question, self.llm)
        return [{
            "question": question,
            "cypher_query": validated_query,
            "explanation": "Generated using text2cypher with refined prompt to retrieve chunk texts and related data."
        }]

# =============================================================================
# FASTAPI ENDPOINTS
# =============================================================================
app = FastAPI(title="Graph RAG API", description="End-to-end Graph Database RAG on Neo4j", version="1.0.0")

@app.post("/upload_documents")
async def upload_documents(files: List[UploadFile] = File(...)):
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

@app.post("/ask_question")
async def ask_question(request: QuestionRequest):
    if request.previous_conversations:
        rewrite_prompt = (
            f"Based on the previous conversation: '{request.previous_conversations}', "
            f"rewrite the query: '{request.question}' into a standalone query."
        )
        rewritten_query = await async_llm.invoke([
            {"role": "system", "content": "You are a professional query rewriting assistant."},
            {"role": "user", "content": rewrite_prompt}
        ])
        logger.debug(f"Rewritten query: {rewritten_query}")
        request.question = rewritten_query.strip()

    cypher_llm = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        model="hf.co/avinashm/text2cypher",
        base_url=OPENAI_BASE_URL,
        temperature=0.2,
        stop=OPENAI_STOP,
        timeout=OPENAI_API_TIMEOUT
    )

    dynamic_schema = get_schema_from_neo4j()
    logger.info(f"Dynamic schema: {dynamic_schema[:200]}...")

    retriever = Text2CypherRetriever(driver, cypher_llm, schema=dynamic_schema)
    generated_queries = await retriever.get_cypher(request.question)

    async def process_query(item: dict) -> dict:
        question_text = item.get("question", "")
        cypher_query = item.get("cypher_query", "")
        query_output = {}
        if cypher_query:
            try:
                query_output["cypher_output"] = await asyncio.to_thread(run_cypher_query, cypher_query)
            except Exception as e:
                query_output["cypher_output"] = {"error": str(e)}
        answer_prompt = (
            f"Given the following data for the question:\n{question_text}\n\n"
            f"Cypher Query:\n{cypher_query}\n\n"
            f"Query Output:\n{json.dumps(query_output.get('cypher_output'), indent=2)}\n\n"
            "Provide a detailed answer based solely on this information. "
            "If no data is returned, explain potential reasons and suggest corrective actions."
        )
        llm_answer = await async_llm.invoke([
            {"role": "system", "content": "You are a professional assistant answering based solely on the provided data."},
            {"role": "user", "content": answer_prompt}
        ])
        return {
            "question": question_text,
            "llm_response": llm_answer.strip(),
            "cypher_query": cypher_query,
            "query_output": query_output.get("cypher_output")
        }

    final_answers = await asyncio.gather(*(process_query(item) for item in generated_queries))
    primary = final_answers[0] if final_answers else {"cypher_query": "", "query_output": {}}
    combined_prompt = build_enhanced_final_prompt(request.question, [a["llm_response"] for a in final_answers],
                                                  primary["cypher_query"], primary.get("query_output", {}))
    combined_llm_response = await async_llm.invoke([
        {"role": "system", "content": "You are a professional assistant synthesizing multiple sources."},
        {"role": "user", "content": combined_prompt}
    ])
    final_response = {
        "user_question": request.question,
        "combined_response": format_natural_response(combined_llm_response.strip()),
        "answers": final_answers
    }
    return final_response

@app.post("/global_search")
async def global_search(request: GlobalSearchRequest):
    answer = await global_search_map_reduce(
        question=request.question,
        conversation_history=request.previous_conversations,
        doc_id=request.doc_id,
        chunk_size=GLOBAL_SEARCH_CHUNK_SIZE,
        top_n=GLOBAL_SEARCH_TOP_N,
        batch_size=GLOBAL_SEARCH_BATCH_SIZE
    )
    logger.debug(f"Raw search answer: {answer}")
    return {"answer": format_natural_response(answer)}

@app.post("/local_search")
async def local_search(request: LocalSearchRequest):
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
        {"role": "system", "content": "You are a professional assistant who answers strictly based on the provided context."},
        {"role": "user", "content": prompt}
    ])

    return {"local_search_answer": answer}

@app.post("/drift_search")
async def drift_search(request: DriftSearchRequest):
    if request.previous_conversations:
        rewrite_prompt = (f"Based on the following conversation history:\n\n{request.previous_conversations}\n\n"
                          f"rewrite the new query '{request.question}' into a standalone, unambiguous query.")
        rewritten_query = await async_llm.invoke([
            {"role": "system", "content": "You are a professional query rewriting assistant."},
            {"role": "user", "content": rewrite_prompt}
        ])
        request.question = rewritten_query.strip()
    summaries = await graph_manager_wrapper.get_stored_community_summaries(doc_id=request.doc_id)
    if not summaries:
        if request.doc_id:
            return {"drift_search_answer": f"No community summaries found for doc_id {request.doc_id}. Please re-index the dataset or try without specifying a doc_id."}
        else:
            return {"drift_search_answer": "No community summaries available. Please re-index the dataset."}
    community_reports = [v["summary"] for v in summaries.values() if v.get("summary")]
    selected_reports = await asyncio.to_thread(select_relevant_communities, request.question, community_reports)
    global_context = "\n\n".join([rep for rep, score in selected_reports])
    conversation_context = f"Conversation History:\n{request.previous_conversations}\n\n" if request.previous_conversations else ""
    primer_prompt = (
        "You are an expert in synthesizing information from diverse community reports. "
        "Using the following community report excerpts:\n\n"
        f"{global_context}\n\n"
        f"{conversation_context}"
        "Answer the following query by generating a hypothetical answer and listing follow-up questions that could "
        "further refine the query."
        "Return your output as a strict JSON object with the keys: 'intermediate_answer', 'follow_up_queries' (a JSON array), and 'score' (a confidence score), and do not include any markdown formatting, code blocks, or extra commentary.\n\n"
        f"Query: {request.question}"
    )
    primer_result = await async_llm.invoke_json([
        {"role": "system", "content": "You are a professional assistant."},
        {"role": "user", "content": primer_prompt}
    ])
    if not isinstance(primer_result, dict) or "intermediate_answer" not in primer_result:
        return {"drift_search_answer": "Primer phase failed to generate a valid response."}
    drift_hierarchy = {"query": request.question, "answer": primer_result["intermediate_answer"],
                       "score": primer_result.get("score", 0), "follow_ups": []}
    follow_up_queries = primer_result.get("follow_up_queries", [])
    for follow_up in follow_up_queries:
        follow_up_text = follow_up if isinstance(follow_up, str) else follow_up.get("query", str(follow_up))
        local_context_text = ""
        local_chunk_query = """
            MATCH (c:Chunk)
            WHERE toLower(c.text) CONTAINS toLower($keyword)
            """
        if request.doc_id:
            local_chunk_query += " AND c.doc_id = $doc_id"
        local_chunk_query += "\nRETURN c.text AS chunk_text\nLIMIT 5"
        params = {"keyword": follow_up_text}
        if request.doc_id:
            params["doc_id"] = request.doc_id
        chunk_results = await asyncio.to_thread(run_cypher_query, local_chunk_query, params)
        if chunk_results:
            chunks = [res.get("chunk_text", "") for res in chunk_results if res.get("chunk_text")]
            local_context_text += "Related Document Chunks:\n" + "\n---\n".join(chunks) + "\n"
        local_conversation = f"Conversation History:\n{request.previous_conversations}\n\n" if request.previous_conversations else ""
        local_prompt = ("You are a professional assistant who refines queries using local document context. "
                        "Based solely on the following local context information:\n\n"
                        f"{local_context_text}\n\n"
                        f"{local_conversation}"
                        "Answer the following follow-up query and, if applicable, propose additional follow-up queries. "
                        "Return your answer as a strict JSON object with keys: 'answer', 'follow_up_queries' (a JSON array), and 'score'.\n\n"
                        f"Follow-Up Query: {follow_up_text}")
        local_result = await async_llm.invoke_json([
            {"role": "system", "content": "You are a professional assistant."},
            {"role": "user", "content": local_prompt}
        ])
        drift_hierarchy["follow_ups"].append(
            {"query": follow_up_text, "answer": local_result.get("answer", ""), "score": local_result.get("score", 0),
             "follow_ups": local_result.get("follow_up_queries", [])})
    reduction_prompt = (
        "You are a professional assistant tasked with synthesizing a final detailed answer from hierarchical query data. "
        "Below is a JSON representation of the query and its follow-up answers:\n\n"
        f"{drift_hierarchy}\n\n"
        "Based solely on the above information, provide a final, comprehensive answer to the original query. "
        "Return your answer as plain text without any additional commentary.\n\n"
        f"Original Query: {request.question}"
    )
    final_answer = await async_llm.invoke([
        {"role": "system", "content": "You are a professional assistant."},
        {"role": "user", "content": reduction_prompt}
    ])
    return {"drift_search_answer": final_answer.strip(), "drift_hierarchy": drift_hierarchy}

@app.get("/documents")
async def list_documents():
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

@app.delete("/delete_document")
async def delete_document(request: DeleteDocumentRequest):
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

@app.get("/communities")
async def get_communities(doc_id: Optional[str] = None):
    try:
        communities = detect_communities(doc_id)
        return {"communities": communities}
    except Exception as e:
        logger.error(f"Error detecting communities: {e}")
        raise HTTPException(status_code=500, detail=f"Error detecting communities: {e}")

@app.get("/community_summaries")
async def community_summaries(doc_id: Optional[str] = None):
    try:
        summaries = await graph_manager_wrapper.get_stored_community_summaries(doc_id)
        return {"community_summaries": summaries}
    except Exception as e:
        logger.error(f"Error generating community summaries: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating community summaries: {e}")

@app.get("/")
async def root():
    return {"message": "GraphRAG API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
