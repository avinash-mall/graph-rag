import os
import re
import json
import string
import uuid
import logging
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
import math
import requests
import blingfire
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Import Text2CypherRetriever from neo4j_graphrag package
from neo4j_graphrag.retrievers.text2cypher import Text2CypherRetriever

# For NER extraction using Flair
from langdetect import detect
from flair.models import SequenceTagger
from flair.data import Sentence
import hashlib
import PyPDF2
import docx
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import spacy
import neuralcoref

# Load spaCy model for NLP and add NeuralCoref
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

# -----------------------------------------------------------------------------
# Load Environment Variables from .env
# -----------------------------------------------------------------------------
load_dotenv()

# Application & Global Configuration
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

# Chunking & Global Search Defaults
CHUNK_SIZE_GDS = int(os.getenv("CHUNK_SIZE_GDS") or "1024")
GLOBAL_SEARCH_CHUNK_SIZE = int(os.getenv("GLOBAL_SEARCH_CHUNK_SIZE") or "1024")
GLOBAL_SEARCH_TOP_N = int(os.getenv("GLOBAL_SEARCH_TOP_N") or "5")
GLOBAL_SEARCH_BATCH_SIZE = int(os.getenv("GLOBAL_SEARCH_BATCH_SIZE") or "20")
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD") or "0.1")

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", "app.log")
# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, LOG_FILE)),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("GraphRAG")
logger.info(f"Logging initialized with level: {LOG_LEVEL}")
llm_logger = logging.getLogger("LLM_API")  # Logger for LLM API calls

# -----------------------------------------------------------------------------
# Neo4j Driver Setup
# -----------------------------------------------------------------------------
driver = GraphDatabase.driver(DB_URL, auth=(DB_USERNAME, DB_PASSWORD))

# -----------------------------------------------------------------------------
# Pydantic Models for Request Bodies (defined only once)
# -----------------------------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str
    doc_id: Optional[Union[str, List[str]]] = None
    previous_conversations: Optional[str] = None

class DeleteDocumentRequest(BaseModel):
    doc_id: Optional[str] = None
    document_name: Optional[str] = None

class LocalSearchRequest(BaseModel):
    entity: str
    question: str

class DriftSearchRequest(BaseModel):
    entity: str
    question: str

# -----------------------------------------------------------------------------
# NER Extraction using Flair (defined only once)
# -----------------------------------------------------------------------------
model_cache = {}
DEFAULT_MODEL_MAP = {
    'en': 'ner-large',
    'de': 'de-ner-large',
    'es': 'es-ner-large',
    'nl': 'nl-ner-large',
    'fr': 'ner-ner',
    'da': 'da-ner',
    'ar': 'ar-ner',
    'uk': 'ner-ukrainian'
}

def extract_flair_entities(text: str) -> List[Dict[str, str]]:
    try:
        lang = detect(text).lower()
    except Exception:
        lang = 'en'
    if '-' in lang:
        lang = lang.split('-')[0]
    model_name = DEFAULT_MODEL_MAP.get(lang, 'ner-large')
    if model_name in model_cache:
        tagger = model_cache[model_name]
    else:
        try:
            tagger = SequenceTagger.load(model_name)
            model_cache[model_name] = tagger
        except Exception as e:
            raise Exception(f"Failed to load model {model_name}: {e}")
    sentence_strs = [s.strip() for s in blingfire.text_to_sentences(text).splitlines() if s.strip()]
    sentences = [Sentence(s) for s in sentence_strs]
    try:
        tagger.predict(sentences, mini_batch_size=32)
    except Exception as e:
        raise Exception(f"NER prediction error: {e}")
    entities = {}
    for sentence in sentences:
        for label in sentence.get_labels():
            key = (label.data_point.text, label.value)
            entities[key] = {"name": label.data_point.text, "label": label.value}
    return list(entities.values())

# -----------------------------------------------------------------------------
# Embedding API Client for Dynamic Community Selection
# -----------------------------------------------------------------------------
class EmbeddingAPIClient:
    def __init__(self):
        self.embedding_api_url = os.getenv("EMBEDDING_API_URL", "https://s-ailabs-gpu5.westeurope.cloudapp.azure.com/api/embed")
        self.timeout = float(os.getenv("API_TIMEOUT", "600"))
        self.logger = logging.getLogger("EmbeddingAPIClient")
        self.logger.setLevel(logging.INFO)
        self.cache = {}  # Local in-memory cache to avoid redundant requests

    def _get_text_hash(self, text: str) -> str:
        """Generate a hash for the input text for deduplication."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get_embedding(self, text: str) -> List[float]:
        text_hash = self._get_text_hash(text)

        if text_hash in self.cache:
            self.logger.info(f"✅ Embedding for text (hash: {text_hash}) retrieved from cache.")
            return self.cache[text_hash]

        self.logger.info("🔍 Requesting embedding for text (first 50 chars): %s", text[:50])

        try:
            response = requests.post(
                self.embedding_api_url,
                json={"model": "mxbai-embed-large", "input": text},
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
                verify=False  # Skipping SSL verification as per request
            )
            response.raise_for_status()

            # Correctly extract embeddings
            embedding = response.json().get("embeddings", [[]])[0]

            if not embedding:
                raise ValueError(f"⚠️ Empty embedding returned for text: {text[:50]}")

            self.logger.debug("✅ Received embedding of length: %d", len(embedding))

            # Cache the embedding to avoid redundant requests
            self.cache[text_hash] = embedding
            return embedding

        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Embedding API error for text (hash: {text_hash}): {e}")
            raise HTTPException(status_code=500, detail="Error in embedding API request")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self.get_embedding(text) for text in texts]



# Instantiate the improved embedding client
embedding_client = EmbeddingAPIClient()

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    from numpy import dot
    from numpy.linalg import norm
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2) + 1e-8)

def select_relevant_communities(
    query: str,
    community_reports: List[Union[str, dict]],
    top_k: int = GLOBAL_SEARCH_TOP_N,
    threshold: float = RELEVANCE_THRESHOLD
) -> List[Tuple[str, float]]:
    query_embedding = embedding_client.get_embedding(query)
    scored_reports = []
    for report in community_reports:
        if isinstance(report, dict) and "embedding" in report:
            report_embedding = report["embedding"]
            summary_text = report["summary"]
        else:
            summary_text = report
            report_embedding = embedding_client.get_embedding(report)
        score = cosine_similarity(query_embedding, report_embedding)
        logger.info("Computed cosine similarity for community report snippet '%s...' is %.4f", summary_text[:50], score)
        if score >= threshold:
            scored_reports.append((summary_text, score))
        else:
            logger.info("Filtered out community report snippet '%s...' with score %.4f (threshold: %.2f)", summary_text[:50], score, threshold)
    scored_reports.sort(key=lambda x: x[1], reverse=True)
    logger.info("Selected community reports after filtering:")
    for rep, score in scored_reports:
        logger.info("Score: %.4f, Snippet: %s", score, rep[:100])
    return scored_reports[:top_k]

def clean_empty_chunks():
    """Removes any chunks in the database that have no text."""
    query = """
    MATCH (c:Chunk)
    WHERE c.text IS NULL OR TRIM(c.text) = ""
    DETACH DELETE c
    """
    try:
        deleted_count = run_cypher_query(query)
        logger.info(f"✅ Cleaned up {len(deleted_count)} empty or invalid chunks from the database.")
    except Exception as e:
        logger.error(f"❌ Error cleaning empty chunks: {e}")

def clean_empty_nodes():
    """Removes any nodes in the database that have no properties or relationships."""
    query = """
    MATCH (n)
    WHERE size(keys(n)) = 0 AND NOT (n)-[]-()
    DELETE n
    """
    try:
        deleted_count = run_cypher_query(query)
        logger.info(f"✅ Cleaned up {len(deleted_count)} empty nodes from the database.")
    except Exception as e:
        logger.error(f"❌ Error cleaning empty nodes: {e}")


def resolve_coreferences(text: str) -> str:
    """Resolves pronouns using NeuralCoref and returns the improved text."""
    try:
        doc = nlp(text)
        resolved_text = doc._.coref_resolved  # Coreference-resolved text
        return resolved_text
    except Exception as e:
        logger.warning(f"Coreference resolution failed: {e}")
        return text  # Return original text in case of errors
        

# -----------------------------------------------------------------------------
# OpenAI LLM Class
# -----------------------------------------------------------------------------
class OpenAI:
    def __init__(self, api_key: str, model: str, base_url: str, temperature: float, stop: List[str]):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.stop = stop

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stop": self.stop
        }
        llm_logger.info("LLM Request Payload: %s", messages)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(self.base_url, json=payload, headers=headers,
                                 timeout=60, verify=False)
        response.raise_for_status()
        output = response.json()["choices"][0]["message"]["content"]
        return output.strip()

# -----------------------------------------------------------------------------
# Utility Functions: JSON Cleaning, Text Processing, and Chunking
# -----------------------------------------------------------------------------
def parse_multiple_json_arrays(response: str) -> List[Dict[str, Any]]:
    arrays = re.findall(r'\[.*?\]', response, re.DOTALL)
    combined = []
    for arr in arrays:
        try:
            data = json.loads(arr)
            if isinstance(data, list):
                combined.extend(data)
        except Exception as e:
            logger.error(f"Error parsing JSON array: {e}. Array: {arr}")
    return combined

def clean_json_response(response: str) -> str:
    response = response.strip()
    if response.startswith("```") and response.endswith("```"):
        lines = response.splitlines()
        response = "\n".join(lines[1:-1])
    return response

def clean_text(text: str) -> str:
    return ''.join(filter(lambda x: x in string.printable, text.strip()))

def chunk_text(text: str, max_chunk_size: int = 512) -> List[str]:
    sentences = [s.strip() for s in blingfire.text_to_sentences(text).splitlines() if s.strip()]
    
    # Deduplicate sentences to avoid repeats
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

# -----------------------------------------------------------------------------
# Neo4j Query Runner and Graph Statistics Functions
# -----------------------------------------------------------------------------
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

# (Additional graph statistics functions omitted for brevity)

# -----------------------------------------------------------------------------
# LLM Prompt Builders for Global Search (Batch Processing)
# -----------------------------------------------------------------------------
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
    prompt += "\nBased on the above responses, provide a final comprehensive answer that directly addresses the original question."
    return prompt.strip()

# -----------------------------------------------------------------------------
# Global Search Map-Reduce Implementation (with Dynamic Community Selection)
# -----------------------------------------------------------------------------
def global_search_map_reduce(question: str, conversation_history: Optional[str] = None,
                             chunk_size: int = GLOBAL_SEARCH_CHUNK_SIZE,
                             top_n: int = GLOBAL_SEARCH_TOP_N,
                             batch_size: int = GLOBAL_SEARCH_BATCH_SIZE) -> str:
    # Retrieve community summaries from Neo4j via the wrapper
    summaries = graph_manager_wrapper.get_stored_community_summaries(doc_id=None)
    if not summaries:
        raise HTTPException(status_code=500, detail="No community summaries available. Please re-index the dataset.")
    
    community_reports = list(summaries.values())
    random.shuffle(community_reports)
    
    # Enhanced Dynamic Community Selection: rank community reports using embeddings
    selected_reports_with_scores = select_relevant_communities(question, community_reports)
    if not selected_reports_with_scores:
        # Fallback: use first top_n reports with default score 0
        selected_reports_with_scores = [(report, 0) for report in community_reports[:top_n]]
    
    # Log the selected community reports and their scores
    logger.info("Selected community reports for dynamic selection:")
    for report, score in selected_reports_with_scores:
        logger.info("Score: %.4f, Snippet: %s", score, report[:100])
    
    llm_instance = OpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        base_url=OPENAI_BASE_URL,
        temperature=OPENAI_TEMPERATURE,
        stop=OPENAI_STOP
    )

    intermediate_points = []
    # Process each selected report
    for report, _ in selected_reports_with_scores:
        chunks = chunk_text(report, max_chunk_size=chunk_size)
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_prompt = (
                "You are an expert in extracting key information. For each of the following community report chunks, "
                "extract key points that are relevant for answering the query. For each chunk, output an object with "
                "'chunk_index' (the index number of the chunk in the batch), 'point' (a short text string), and 'rating' "
                "(a numerical value between 1 and 100 indicating its importance). "
                "Return only a valid JSON array (with no extra commentary)."
            )
            for idx, chunk in enumerate(batch):
                batch_prompt += f"\n\nChunk {idx}:\n\"\"\"\n{chunk}\n\"\"\""
            batch_prompt += f"\n\nUser Query:\n\"\"\"\n{question}\n\"\"\""
            try:
                response = llm_instance.invoke([
                    {"role": "system", "content": "You are a professional extraction assistant who strictly follows user's request."},
                    {"role": "user", "content": batch_prompt}
                ])
                cleaned_response = clean_json_response(response)
                points = parse_multiple_json_arrays(cleaned_response)
                intermediate_points.extend(points)
            except Exception as e:
                logger.error(f"Batch processing error: {e}.")
    
    # Log all intermediate key points before sorting
    logger.info("Intermediate key points extracted (unsorted):")
    for pt in intermediate_points:
        logger.info("Key point: %s, Rating: %s", pt.get("point"), pt.get("rating"))
    
    # Sort intermediate points by rating (default missing rating to 0)
    intermediate_points_sorted = sorted(intermediate_points, key=lambda x: x.get("rating") or 0, reverse=True)
    selected_points = intermediate_points_sorted[:top_n]
    aggregated_context = "\n".join([f"{pt['point']} (Rating: {pt['rating']})" for pt in selected_points])
    
    conv_text = f"Conversation History: {conversation_history}\n" if conversation_history else ""
    reduce_prompt = f"""
You are a professional synthesizer answering "User Query" from the provided "Aggregated Key Points". Format your final answer strictly as a raw JSON object (without any markdown code fences) with the following keys:
  "summary": a detailed overall summary,
  "key_points": a list of key points (each an object with keys "text" and "importance"),
  "additional_notes": any extra relevant observations.

Aggregated Key Points:
{aggregated_context}

User Query:
{question}
""".strip()
    
    try:
        final_answer = llm_instance.invoke([
            {"role": "system", "content": "You are a professional assistant skilled in synthesizing information who strictly follows user's request."},
            {"role": "user", "content": reduce_prompt}
        ])
    except Exception as e:
        final_answer = f"Error during reduce step: {e}"
    return final_answer

# -----------------------------------------------------------------------------
# Graph Manager Class (Building & Projecting the Graph)
# -----------------------------------------------------------------------------
class GraphManager:
    def __init__(self):
        pass

    def _merge_entity(self, session, name: str, doc_id: str, chunk_id: int):
        query = """
        MERGE (e:Entity {name: $name, doc_id: $doc_id})
        MERGE (e)-[:MENTIONED_IN]->(c:Chunk {id: $cid, doc_id: $doc_id})
        """
        session.run(query, name=name, doc_id=doc_id, cid=chunk_id)

    def _merge_cooccurrence(self, session, name_a: str, name_b: str, doc_id: str):
        query = """
            MATCH (a:Entity {name: $name_a, doc_id: $doc_id})-[:MENTIONED_IN]->(c:Chunk)
            MATCH (b:Entity {name: $name_b, doc_id: $doc_id})-[:MENTIONED_IN]->(c)
            MERGE (entity_a:Entity {name: $name_a, doc_id: $doc_id})
            MERGE (entity_b:Entity {name: $name_b, doc_id: $doc_id})
            ON CREATE SET entity_a.created_at = timestamp()
            ON CREATE SET entity_b.created_at = timestamp()
            MERGE (entity_a)-[:CO_OCCURS_WITH]->(entity_b)
        """
        session.run(query, name_a=name_a, name_b=name_b, doc_id=doc_id)

    def build_graph(self, chunks: List[str], metadata_list: List[Dict[str, Any]]) -> None:
        with driver.session() as session:
            for i, (chunk, meta) in enumerate(zip(chunks, metadata_list)):
                if not chunk.strip() or chunk.strip().isdigit():
                    logger.warning(f"Skipping empty or numeric chunk with ID {i}")
                    continue
    
                # Compute the embedding for the chunk text
                try:
                    chunk_embedding = embedding_client.get_embedding(chunk)
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
                session.run(
                    query,
                    cid=i,
                    doc_id=meta["doc_id"],
                    text=chunk,
                    document_name=meta.get("document_name"),
                    timestamp=meta.get("timestamp"),
                    embedding=chunk_embedding
                )
                logger.info(f"Added chunk {i} with text: {chunk[:100]}...")
                try:
                    entities = extract_flair_entities(chunk)
                except Exception as e:
                    logger.error(f"NER extraction error on chunk {i}: {e}")
                    entities = []
                names = [e["name"].strip().lower() for e in entities]
                for name in names:
                    self._merge_entity(session, name, meta["doc_id"], i)
                if len(names) > 1:
                    for j in range(len(names)):
                        for k in range(j + 1, len(names)):
                            self._merge_cooccurrence(session, names[j], names[k], meta["doc_id"])
                try:
                    client = OpenAI(
                        api_key=OPENAI_API_KEY,
                        model=OPENAI_MODEL,
                        base_url=OPENAI_BASE_URL,
                        temperature=OPENAI_TEMPERATURE,
                        stop=OPENAI_STOP
                    )
                    summary = client.invoke([
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
                            rel_query = """
                                MERGE (a:Entity {name: $source, doc_id: $doc_id})
                                MERGE (b:Entity {name: $target, doc_id: $doc_id})
                                MERGE (a)-[r:RELATES_TO {weight: $weight}]->(b)
                                WITH a, b
                                MATCH (c:Chunk {id: $cid, doc_id: $doc_id})
                                MERGE (a)-[:MENTIONED_IN]->(c)
                                MERGE (b)-[:MENTIONED_IN]->(c)
                            """
                            session.run(rel_query, source=source, target=target, weight=weight, doc_id=meta["doc_id"], cid=i)

        logger.info("Graph construction complete.")

    def reproject_graph(self, graph_name: str = GRAPH_NAME, doc_id: Optional[str] = None) -> None:
        with driver.session() as session:
            exists_record = session.run("CALL gds.graph.exists($graph_name) YIELD exists",
                                        graph_name=graph_name).single()
            if exists_record and exists_record["exists"]:
                session.run("CALL gds.graph.drop($graph_name)", graph_name=graph_name)
            if doc_id:
                session.run("MATCH (n:Entity {doc_id: $doc_id}) SET n:DocEntity", doc_id=doc_id)
                config = {"RELATES_TO": {"orientation": "UNDIRECTED"}}
                session.run(
                    "CALL gds.graph.project($graph_name, ['DocEntity'], $config) YIELD graphName",
                    graph_name=graph_name, config=config
                )
                session.run("MATCH (n:DocEntity {doc_id: $doc_id}) REMOVE n:DocEntity", doc_id=doc_id)
            else:
                config = {"RELATES_TO": {"orientation": "UNDIRECTED"}}
                session.run(
                    "CALL gds.graph.project($graph_name, ['Entity'], $config)",
                    graph_name=graph_name, config=config
                )
            logger.info("Graph projection complete.")

# -----------------------------------------------------------------------------
# Instantiate GraphManager
# -----------------------------------------------------------------------------
graph_manager = GraphManager()

# -----------------------------------------------------------------------------
# Global Function: detect_communities
# -----------------------------------------------------------------------------
def detect_communities(doc_id: Optional[str] = None) -> dict:
    communities = {}
    with driver.session() as session:
        exists_record = session.run("CALL gds.graph.exists($graph_name) YIELD exists", graph_name=GRAPH_NAME).single()
        if exists_record and exists_record["exists"]:
            session.run("CALL gds.graph.drop($graph_name)", graph_name=GRAPH_NAME)
        if doc_id:
            session.run("MATCH (n:Entity {doc_id: $doc_id}) SET n:DocEntity", doc_id=doc_id)
            config = {"RELATES_TO": {"orientation": "UNDIRECTED"}}
            session.run(
                "CALL gds.graph.project($graph_name, ['DocEntity'], $config) YIELD graphName",
                graph_name=GRAPH_NAME, config=config
            )
        else:
            config = {"RELATES_TO": {"orientation": "UNDIRECTED"}}
            session.run(
                "CALL gds.graph.project($graph_name, ['Entity'], $config)",
                graph_name=GRAPH_NAME, config=config
            )
        result = session.run(
            "CALL gds.leiden.stream($graph_name) YIELD nodeId, communityId "
            "RETURN gds.util.asNode(nodeId).name AS entity, communityId AS community ORDER BY community, entity",
            graph_name=GRAPH_NAME
        )
        for record in result:
            comm = record["community"]
            entity = record["entity"]
            communities.setdefault(comm, []).append(entity)
        session.run("CALL gds.graph.drop($graph_name)", graph_name=GRAPH_NAME)
        if doc_id:
            session.run("MATCH (n:DocEntity {doc_id: $doc_id}) REMOVE n:DocEntity", doc_id=doc_id)
    return communities

def generate_community_summaries(doc_id: Optional[str] = None) -> dict:
    communities = detect_communities(doc_id)
    llm = OpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        base_url=OPENAI_BASE_URL,
        temperature=OPENAI_TEMPERATURE,
        stop=OPENAI_STOP
    )
    summaries = {}
    for comm, entities in communities.items():
        prompt = f"Summarize the following community of entities: {', '.join(entities)}. Provide a concise summary capturing the main themes and key relationships."
        summary = llm.invoke([
            {"role": "system", "content": "You are a professional summarization assistant."},
            {"role": "user", "content": prompt}
        ])
        summaries[comm] = summary
    return summaries

# -----------------------------------------------------------------------------
# Graph Manager Extended: Storing and Retrieving Community Summaries
# -----------------------------------------------------------------------------
class GraphManagerExtended:
    def __init__(self, driver):
        self.driver = driver

    def store_community_summaries(self, doc_id: str) -> None:
        with self.driver.session() as session:
            communities = detect_communities(doc_id)
            logger.debug(f"Detected communities for doc_id {doc_id}: {communities}")
            llm = OpenAI(
                api_key=OPENAI_API_KEY,
                model=OPENAI_MODEL,
                base_url=OPENAI_BASE_URL,
                temperature=OPENAI_TEMPERATURE,
                stop=OPENAI_STOP
            )
            if not communities:
                logger.warning(f"No communities detected for doc_id {doc_id}. Creating default summary.")
                default_summary = "No communities detected."
                default_embedding = embedding_client.get_embedding(default_summary)
                store_query = """
                CREATE (cs:CommunitySummary {doc_id: $doc_id, community: 'default', summary: $summary, embedding: $embedding, timestamp: $timestamp})
                """
                session.run(
                    store_query,
                    doc_id=doc_id,
                    summary=default_summary,
                    embedding=default_embedding,
                    timestamp=datetime.now().isoformat()
                )
            else:
                for comm, entities in communities.items():
                    chunk_query = """
                    MATCH (e:Entity {doc_id: $doc_id})-[:MENTIONED_IN]->(c:Chunk)
                    WHERE toLower(e.name) IN $entity_names AND c.text IS NOT NULL AND c.text <> ""
                    WITH collect(DISTINCT c.text) AS texts
                    RETURN apoc.text.join(texts, "\n") AS aggregatedText
                    """
                    
                    result = session.run(
                        chunk_query,
                        doc_id=doc_id,
                        entity_names=[e.lower() for e in entities]
                    )
                    record = result.single()
            
                    # Ensure 'aggregated_text' is always defined
                    aggregated_text = record["aggregatedText"] if record and record["aggregatedText"] else ""
                    
                    # Handle cases where no content is found
                    if not aggregated_text.strip():
                        logger.warning(f"No content found for community {comm}. Using entity names as fallback.")
                        aggregated_text = ", ".join(entities)
            
                    # Improved Prompt Formatting
                    prompt = (
                        f"Summarize the following text into a concise summary that highlights the main themes and key relationships. "
                        f"Do not include any extra commentary about the text's structure or quality.\n\n"
                        f"{aggregated_text}"
                    )
            
                    logger.info(f"Aggregated text for community {comm}: {aggregated_text}")
                    
                    try:
                        summary = llm.invoke([
                            {"role": "system", "content": "You are a professional summarization assistant."},
                            {"role": "user", "content": prompt}
                        ])
                    except Exception as e:
                        logger.error(f"Failed to generate summary for community {comm}: {e}")
                        summary = "No summary available due to error."
            
                    logger.debug(f"Storing summary for community {comm}: {summary}")
            
                    try:
                        summary_embedding = embedding_client.get_embedding(summary)
                    except Exception as e:
                        logger.error(f"Failed to generate embedding for summary of community {comm}: {e}")
                        summary_embedding = []
            
                    # Ensure the community summary is stored safely
                    store_query = """
                    MERGE (cs:CommunitySummary {doc_id: $doc_id, community: $community})
                    SET cs.summary = $summary, cs.embedding = $embedding, cs.timestamp = $timestamp
                    """
                    session.run(
                        store_query,
                        doc_id=doc_id,
                        community=comm,
                        summary=summary,
                        embedding=summary_embedding,
                        timestamp=datetime.now().isoformat()
                    )


    def get_stored_community_summaries(self, doc_id: Optional[str] = None) -> dict:
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

# -----------------------------------------------------------------------------
# Graph Manager Wrapper
# -----------------------------------------------------------------------------
class GraphManagerWrapper:
    def __init__(self):
        self.manager = graph_manager  # global graph_manager instance
    
    def build_graph(self, chunks: List[str], metadata_list: List[Dict[str, Any]]) -> None:
        self.manager.build_graph(chunks, metadata_list)
    
    def reproject_graph(self, graph_name: str = GRAPH_NAME, doc_id: Optional[str] = None) -> None:
        self.manager.reproject_graph(graph_name, doc_id)
    
    def store_community_summaries(self, doc_id: str) -> None:
        extended = GraphManagerExtended(driver)
        extended.store_community_summaries(doc_id)
    
    def get_stored_community_summaries(self, doc_id: Optional[str] = None) -> dict:
        extended = GraphManagerExtended(driver)
        return extended.get_stored_community_summaries(doc_id)

graph_manager_wrapper = GraphManagerWrapper()

# -----------------------------------------------------------------------------
# FastAPI Application and Endpoints
# -----------------------------------------------------------------------------
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

        text = clean_text(text)   # Step 1: Clean text
        text = resolve_coreferences(text)  # Step 2: Coreference resolution
        
        doc_id = str(uuid.uuid4())
        metadata.append({
            "doc_id": doc_id,
            "document_name": file.filename,
            "timestamp": datetime.now().isoformat()
        })
        document_texts.append(text)

    all_chunks = []
    meta_list = []
    for text, meta in zip(document_texts, metadata):
        chunks = chunk_text(text, max_chunk_size=CHUNK_SIZE_GDS)
        logger.info(f"Generated {len(chunks)} chunks for document {meta['document_name']}")
        all_chunks.extend(chunks)
        meta_list.extend([meta] * len(chunks))

    try:
        graph_manager.build_graph(all_chunks, meta_list)
    except Exception as e:
        logger.error(f"Graph building error: {e}")
        raise HTTPException(status_code=500, detail=f"Graph building error: {e}")

    # Cleanup logic for empty chunks and nodes
    clean_empty_chunks()
    clean_empty_nodes()

    unique_doc_ids = set(meta["doc_id"] for meta in metadata)
    for doc_id in unique_doc_ids:
        try:
            graph_manager_wrapper.store_community_summaries(doc_id)
        except Exception as e:
            logger.error(f"Error storing community summaries for doc_id {doc_id}: {e}")

    return {"message": "Documents processed, graph updated, and community summaries stored successfully."}


@app.post("/ask_question")
def ask_question(request: QuestionRequest):
    llm_instance = OpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        base_url=OPENAI_BASE_URL,
        temperature=OPENAI_TEMPERATURE,
        stop=OPENAI_STOP
    )
    if request.previous_conversations:
        rewrite_prompt = (
            f"Based on the user's previous history query about '{request.previous_conversations}', "
            f"rewrite their new query: '{request.question}' into a standalone query."
        )
        rewritten_query = llm_instance.invoke([
            {"role": "system", "content": "You are a professional query rewriting assistant."},
            {"role": "user", "content": rewrite_prompt}
        ])
        logger.debug(f"Rewritten query: {rewritten_query}")
        request.question = rewritten_query
    retriever = Text2CypherRetriever(driver, llm_instance)
    try:
        generated_queries = retriever.get_cypher(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating cypher queries: {e}")
    final_answers = []
    for item in generated_queries:
        question_text = item.get("question", "")
        query_types = {
            "standard_query": item.get("standard_query", ""),
            "fuzzy_query": item.get("fuzzy_query", ""),
            "general_query": item.get("general_query", "")
        }
        query_outputs = {}
        for key, query in query_types.items():
            if query:
                try:
                    query_outputs[f"{key}_output"] = run_cypher_query(query)
                except Exception as e:
                    query_outputs[f"{key}_output"] = {"error": str(e)}
            else:
                query_outputs[f"{key}_output"] = []
        query_context = {
            "question": question_text,
            "standard_query": query_types["standard_query"],
            "standard_query_output": query_outputs["standard_query_output"],
            "fuzzy_query": query_types["fuzzy_query"],
            "fuzzy_query_output": query_outputs["fuzzy_query_output"],
            "general_query": query_types["general_query"],
            "general_query_output": query_outputs["general_query_output"],
            "explanation": item.get("explanation", "")
        }
        answer_prompt = build_llm_answer_prompt(query_context)
        logger.debug(f"Answer prompt for question '{question_text}':\n{answer_prompt}")
        try:
            llm_answer = llm_instance.invoke([
                {"role": "system",
                 "content": "You are a professional assistant who answers queries concisely based solely on the provided Neo4j Cypher query outputs."},
                {"role": "user", "content": answer_prompt}
            ])
        except Exception as e:
            llm_answer = f"LLM error: {e}"
        final_answers.append({
            "question": question_text,
            "llm_response": llm_answer.strip()
        })
    responses_for_combined = [answer["llm_response"] for answer in final_answers]
    combined_prompt = build_combined_answer_prompt(request.question, responses_for_combined)
    logger.debug(f"Combined prompt:\n{combined_prompt}")
    try:
        combined_llm_response = llm_instance.invoke([
            {"role": "system",
             "content": "You are a professional assistant who synthesizes information from multiple sources to provide a detailed final answer."},
            {"role": "user", "content": combined_prompt}
        ])
    except Exception as e:
        combined_llm_response = f"LLM error: {e}"
    final_response = {
        "user_question": request.question,
        "combined_response": combined_llm_response.strip(),
        "answers": final_answers
    }
    logger.debug(f"Final response: {json.dumps(final_response, indent=2)}")
    return final_response

@app.post("/global_search")
def global_search(request: QuestionRequest):
    answer = global_search_map_reduce(
        question=request.question,
        conversation_history=request.previous_conversations,
        chunk_size=GLOBAL_SEARCH_CHUNK_SIZE,
        top_n=GLOBAL_SEARCH_TOP_N,
        batch_size=GLOBAL_SEARCH_BATCH_SIZE
    )
    return {"global_search_answer": answer}

@app.post("/local_search")
def local_search(request: LocalSearchRequest):
    llm_instance = OpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        base_url=OPENAI_BASE_URL,
        temperature=OPENAI_TEMPERATURE,
        stop=OPENAI_STOP
    )
    query = """
    MATCH (e:Entity {name: $entity})
    OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
    RETURN e.name AS entity, collect({neighbor: neighbor.name, relationship: type(r)}) as neighbors
    """
    result = run_cypher_query(query, {"entity": request.entity.lower()})
    if result:
        row = result[0]
        context = f"Entity: {row['entity']}\n"
        if row.get('neighbors'):
            neighbors_list = [f"{item['neighbor']} ({item.get('relationship', '')})" for item in row['neighbors'] if item.get('neighbor')]
            context += "Neighbors: " + ", ".join(neighbors_list) + "\n"
        else:
            context += "No neighbors found.\n"
    else:
        context = "No local context found."
    prompt = f"Based on the following local context:\n{context}\n\nAnswer the following question about the entity '{request.entity}': {request.question}"
    try:
        answer = llm_instance.invoke([
            {"role": "system", "content": "You are a professional assistant skilled in reasoning based on local context information."},
            {"role": "user", "content": prompt}
        ])
    except Exception as e:
        answer = f"LLM error: {e}"
    return {"local_search_answer": answer}

@app.post("/drift_search")
def drift_search(request: DriftSearchRequest):
    llm_instance = OpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        base_url=OPENAI_BASE_URL,
        temperature=OPENAI_TEMPERATURE,
        stop=OPENAI_STOP
    )
    local_context_text = ""
    query = """
    MATCH (e:Entity {name: $entity})
    OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
    RETURN e.name AS entity, collect({neighbor: neighbor.name, relationship: type(r)}) as neighbors
    """
    result = run_cypher_query(query, {"entity": request.entity.lower()})
    if result:
        row = result[0]
        local_context_text = f"Entity: {row['entity']}\n"
        if row.get('neighbors'):
            neighbors_list = [f"{item['neighbor']} ({item.get('relationship', '')})" for item in row['neighbors'] if item.get('neighbor')]
            local_context_text += "Neighbors: " + ", ".join(neighbors_list) + "\n"
        else:
            local_context_text += "No neighbors found.\n"
    community_context_text = ""
    communities = detect_communities(doc_id=None)
    community_for_entity = None
    for comm, entities in communities.items():
        if request.entity.lower() in [e.lower() for e in entities]:
            community_for_entity = comm
            break
    if community_for_entity is not None:
        summaries = graph_manager_wrapper.get_stored_community_summaries(doc_id=None)
        summary = summaries.get(community_for_entity, "No summary available.")
        community_context_text = f"Community {community_for_entity} summary: {summary}"
    else:
        community_context_text = "Entity not found in any community."
    prompt = (
        f"Based on the following information:\nLocal Context:\n{local_context_text}\n\n"
        f"Community Context:\n{community_context_text}\n\n"
        f"Answer the following question about the entity '{request.entity}': {request.question}"
    )
    try:
        answer = llm_instance.invoke([
            {"role": "system", "content": "You are a professional assistant skilled in synthesizing local and community context to provide detailed answers."},
            {"role": "user", "content": prompt}
        ])
    except Exception as e:
        answer = f"LLM error: {e}"
    return {"drift_search_answer": answer}

@app.get("/documents")
def list_documents():
    try:
        query = """
            MATCH (c:Chunk)
            RETURN DISTINCT c.doc_id AS doc_id, c.document_name AS document_name, c.timestamp AS timestamp
        """
        results = run_cypher_query(query)
        return {"documents": results}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Error listing documents")

@app.delete("/delete_document")
def delete_document(request: DeleteDocumentRequest):
    if not request.doc_id and not request.document_name:
        raise HTTPException(status_code=400, detail="Provide either doc_id or document_name to delete a document.")
    try:
        if request.doc_id:
            query = "MATCH (n) WHERE n.doc_id = $doc_id DETACH DELETE n"
            run_cypher_query(query, {"doc_id": request.doc_id})
            message = f"Document with doc_id {request.doc_id} deleted successfully."
        else:
            query = "MATCH (n) WHERE n.document_name = $document_name DETACH DELETE n"
            run_cypher_query(query, {"document_name": request.document_name})
            message = f"Document with name {request.document_name} deleted successfully."
        return {"message": message}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {e}")

@app.get("/communities")
def get_communities(doc_id: Optional[str] = None):
    try:
        communities = detect_communities(doc_id)
        return {"communities": communities}
    except Exception as e:
        logger.error(f"Error detecting communities: {e}")
        raise HTTPException(status_code=500, detail=f"Error detecting communities: {e}")

@app.get("/community_summaries")
def community_summaries(doc_id: Optional[str] = None):
    try:
        summaries = graph_manager_wrapper.get_stored_community_summaries(doc_id)
        return {"community_summaries": summaries}
    except Exception as e:
        logger.error(f"Error generating community summaries: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating community summaries: {e}")

@app.get("/")
def root():
    return {"message": "GraphRAG API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
