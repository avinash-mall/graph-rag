from __future__ import annotations
import asyncio
import logging
import os
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

import PyPDF2
import docx
import urllib3
import httpx
import blingfire
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, UploadFile, File
from neo4j import GraphDatabase
from pydantic import BaseModel

from utils import run_async, clean_text, chunk_text, run_cypher_query, extract_entities_with_llm

# Disable warnings and load environment variables
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

# Global Configuration
APP_HOST = os.getenv("APP_HOST") or "0.0.0.0"
APP_PORT = int(os.getenv("APP_PORT") or "8000")
GRAPH_NAME = os.getenv("GRAPH_NAME")
DB_URL = os.getenv("DB_URL")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
CHUNK_SIZE_GDS = int(os.getenv("CHUNK_SIZE_GDS") or "512")
COREF_WORD_LIMIT = int(os.getenv("COREF_WORD_LIMIT", "8000"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "llama3.2"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE") or "0.0")
OPENAI_STOP = os.getenv("OPENAI_STOP")
if OPENAI_STOP:
    import json
    OPENAI_STOP = json.loads(OPENAI_STOP)
API_TIMEOUT = int(os.getenv("API_TIMEOUT") or "600")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "mxbai-embed-large")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
COREF_SYSTEM_PROMPT = os.getenv("COREF_SYSTEM_PROMPT")
COREF_USER_PROMPT = os.getenv("COREF_USER_PROMPT")
SUMMARY_SYSTEM_PROMPT = os.getenv("SUMMARY_SYSTEM_PROMPT")
SUMMARY_USER_PROMPT = os.getenv("SUMMARY_USER_PROMPT")
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("DocumentAPI")

# Initialize Neo4j driver
driver = GraphDatabase.driver(DB_URL, auth=(DB_USERNAME, DB_PASSWORD))

# Pydantic model for document deletion
class DeleteDocumentRequest(BaseModel):
    doc_id: Optional[str] = None
    document_name: Optional[str] = None

# ----------------------------
# Utility Functions Specific to Document Processing
# ----------------------------

# Asynchronous Community Detection Function
async def detect_communities(doc_id: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Asynchronously detect communities using the Neo4j Graph Data Science Leiden algorithm.
    If a doc_id is provided, only entities for that document are used.
    """
    def _detect():
        communities = {}
        with driver.session() as session:
            # If a graph projection already exists, drop it
            exists_record = session.run("CALL gds.graph.exists($graph_name) YIELD exists", graph_name=GRAPH_NAME).single()
            if exists_record and exists_record["exists"]:
                session.run("CALL gds.graph.drop($graph_name)", graph_name=GRAPH_NAME)
            # Project the graph – if a doc_id is provided, tag the relevant entities with a label.
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
            # Run the Leiden algorithm and collect communities.
            result = session.run(
                "CALL gds.leiden.stream($graph_name) YIELD nodeId, communityId "
                "RETURN gds.util.asNode(nodeId).name AS entity, communityId AS community "
                "ORDER BY community, entity",
                graph_name=GRAPH_NAME
            )
            for record in result:
                comm = record["community"]
                entity = record["entity"]
                communities.setdefault(str(comm), []).append(entity)
            # Clean up: drop the projected graph and remove the temporary label.
            session.run("CALL gds.graph.drop($graph_name)", graph_name=GRAPH_NAME)
            if doc_id:
                session.run("MATCH (n:DocEntity {doc_id: $doc_id}) REMOVE n:DocEntity", doc_id=doc_id)
        return communities

    communities = await asyncio.to_thread(_detect)
    return communities

def clean_empty_chunks():
    """
    Delete Chunk nodes in Neo4j with empty or null 'text' property.
    """
    with driver.session() as session:
        query = "MATCH (c:Chunk) WHERE c.text IS NULL OR trim(c.text) = '' DETACH DELETE c"
        session.run(query)
        logger.info("Cleaned empty Chunk nodes.")

def clean_empty_nodes():
    """
    Delete Entity nodes in Neo4j with empty or null 'name' property.
    """
    with driver.session() as session:
        query = "MATCH (n:Entity) WHERE n.name IS NULL OR trim(n.name) = '' DETACH DELETE n"
        session.run(query)
        logger.info("Cleaned empty Entity nodes.")

async def resolve_coreferences_in_parts(text: str) -> str:
    """
    Resolve ambiguous pronouns in text using the LLM.
    Splits text into parts based on COREF_WORD_LIMIT, resolves each, then recombines.
    """
    words = text.split()
    parts = [" ".join(words[i:i + COREF_WORD_LIMIT]) for i in range(0, len(words), COREF_WORD_LIMIT)]

    async def process_part(part: str, idx: int) -> str:
        prompt = COREF_USER_PROMPT + part
        try:
            resolved_text = await async_llm.invoke([
                {"role": "system", "content": COREF_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ])
            logger.info(f"Coreference resolution successful for part {idx + 1}/{len(parts)}.")
            return resolved_text.strip()
        except Exception as e:
            logger.warning(f"Coreference resolution failed for part {idx + 1}/{len(parts)}: {e}")
            return part

    resolved_parts = await asyncio.gather(*[process_part(part, idx) for idx, part in enumerate(parts)])
    combined_text = " ".join(resolved_parts)
    filtered_text = re.sub(
        r'\b(he|she|it|they|him|her|them|his|her|their|its|himself|herself|his partner)\b',
        '',
        combined_text,
        flags=re.IGNORECASE
    )
    return filtered_text.strip()


def get_refined_system_message() -> str:
    """
    Return a refined system message instructing the LLM to return valid JSON.
    """
    return "You are a professional assistant. Please provide a detailed answer."

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
        logger.debug(f"LLM Payload Sent: {payload}")
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
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    async def get_embedding(self, text: str) -> List[float]:
        text_hash = self._get_text_hash(text)
        if text_hash in self.cache:
            self.logger.info(f"Embedding for text (hash: {text_hash}) retrieved from cache.")
            return self.cache[text_hash]
        self.logger.info("Requesting embedding for text (first 50 chars): %s", text[:50])
        payload = {"model": EMBEDDING_MODEL_NAME, "input": text}
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
# Graph Manager: Build Graph and Store Community Summaries
# ----------------------------
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
                # Merge the Chunk node
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
                # Extract entities using the new NER function
                entities = await extract_entities_with_llm(chunk)
                logger.info(f"Extracted entities from chunk {i}: {entities}")
                # Create Entity nodes and link them to the Chunk node
                for entity in entities:
                    entity_name = entity.get("name", "").strip().lower()
                    entity_type = entity.get("type", "").strip()
                    if not entity_name:
                        continue
                    try:
                        entity_embedding = await async_embedding_client.get_embedding(entity_name)
                    except Exception as e:
                        logger.error(f"Error generating embedding for entity {entity_name}: {e}")
                        continue
                    entity_query = """
                        MERGE (e:Entity {name: $name, doc_id: $doc_id})
                        ON CREATE SET e.embedding = $embedding, e.type = $type
                        ON MATCH SET e.embedding = $embedding, e.type = $type
                    """
                    session.run(entity_query, name=entity_name, doc_id=meta["doc_id"],
                                embedding=entity_embedding, type=entity_type)
                    link_query = """
                        MATCH (e:Entity {name: $name, doc_id: $doc_id})
                        MATCH (c:Chunk {id: $cid, doc_id: $doc_id})
                        MERGE (e)-[:MENTIONED_IN]->(c)
                    """
                    session.run(link_query, name=entity_name, doc_id=meta["doc_id"], cid=i)
                # Create relationships between all pairs of entities extracted from the same chunk
                if len(entities) > 1:
                    for j in range(len(entities)):
                        for k in range(j+1, len(entities)):
                            source = entities[j].get("name", "").strip().lower()
                            target = entities[k].get("name", "").strip().lower()
                            if not source or not target:
                                continue
                            rel_query = """
                                MATCH (a:Entity {name: $source, doc_id: $doc_id})
                                MATCH (b:Entity {name: $target, doc_id: $doc_id})
                                MERGE (a)-[r:RELATES_TO]->(b)
                                SET r.weight = $weight
                                WITH a, b
                                MATCH (c:Chunk {id: $cid, doc_id: $doc_id})
                                MERGE (a)-[:MENTIONED_IN]->(c)
                                MERGE (b)-[:MENTIONED_IN]->(c)
                            """
                            session.run(rel_query, source=source, target=target, doc_id=meta["doc_id"], cid=i, weight=1.0)
        logger.info("Graph construction complete.")

    async def store_community_summaries(self, doc_id: str) -> None:
        communities = await detect_communities(doc_id)
        logger.debug(f"detect_communities returned: {communities}")
        if not communities:
            communities = {"default": ["No communities detected."]}

        with driver.session() as session:
            for comm, entities in communities.items():
                chunk_query = """
                    MATCH (e:Entity {doc_id: $doc_id})-[:MENTIONED_IN]->(c:Chunk)
                    WHERE toLower(e.name) IN $entity_names AND c.text IS NOT NULL AND c.text <> ""
                    RETURN collect(DISTINCT c.text) AS texts
                """
                result = session.run(
                    chunk_query,
                    doc_id=doc_id,
                    entity_names=[e.lower() for e in entities]
                )
                record = result.single()
                texts = record["texts"] if record and record["texts"] else []
                aggregated_text = "\n".join(texts)
                if not aggregated_text.strip():
                    logger.warning(f"No content found for community {comm}. Skipping community summary creation.")
                    continue

                prompt = SUMMARY_USER_PROMPT + aggregated_text
                summary = await async_llm.invoke([
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ])

                try:
                    summary_embedding = await async_embedding_client.get_embedding(summary)
                except Exception as e:
                    logger.error(f"Failed to generate embedding for summary of community {comm}: {e}")
                    summary_embedding = []

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
        logger.info("Stored community summaries.")


class GraphManagerWrapper:
    def __init__(self):
        self.manager = GraphManager()

    async def build_graph(self, chunks: List[str], metadata_list: List[Dict[str, Any]]) -> None:
        await self.manager.build_graph(chunks, metadata_list)

    async def store_community_summaries(self, doc_id: str) -> None:
        await self.manager.store_community_summaries(doc_id)

graph_manager_wrapper = GraphManagerWrapper()

# ----------------------------
# API Router for Document-Related Endpoints
# ----------------------------
router = APIRouter()

@router.post("/upload_documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload documents, process them into text chunks, build the graph, and store community summaries.
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
        # Resolve ambiguous references using LLM coreference resolution.
        text = await resolve_coreferences_in_parts(text)
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
        await graph_manager_wrapper.build_graph(all_chunks, meta_list)
    except Exception as e:
        logger.error(f"Graph building error: {e}")
        raise HTTPException(status_code=500, detail=f"Graph building error: {e}")
    # Clean up any empty nodes
    clean_empty_chunks()
    clean_empty_nodes()
    unique_doc_ids = {meta["doc_id"] for meta in metadata}
    for doc_id in unique_doc_ids:
        try:
            await graph_manager_wrapper.store_community_summaries(doc_id)
        except Exception as e:
            logger.error(f"Error storing community summaries for doc_id {doc_id}: {e}")
    return {"message": "Documents processed, graph updated, and community summaries stored successfully."}

@router.get("/documents")
async def list_documents():
    """
    List all distinct documents based on chunks in the graph.
    """
    try:
        query = """
            MATCH (c:Chunk)
            RETURN DISTINCT c.doc_id AS doc_id, c.document_name AS document_name, c.timestamp AS timestamp
        """
        documents = run_cypher_query(driver, query)
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Error listing documents")

@router.delete("/delete_document")
async def delete_document(request: DeleteDocumentRequest):
    """
    Delete a document from the graph based on doc_id or document_name.
    """
    if not request.doc_id and not request.document_name:
        raise HTTPException(status_code=400, detail="Provide either doc_id or document_name to delete a document.")
    try:
        with driver.session() as session:
            if request.doc_id:
                query = "MATCH (n) WHERE n.doc_id = $doc_id DETACH DELETE n"
                session.run(query, doc_id=request.doc_id)
                message = f"Document with doc_id {request.doc_id} deleted successfully."
            else:
                query = "MATCH (n) WHERE n.document_name = $document_name DETACH DELETE n"
                session.run(query, document_name=request.document_name)
                message = f"Document with name {request.document_name} deleted successfully."
        return {"message": message}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {e}")

@router.get("/communities")
async def get_communities(doc_id: Optional[str] = None):
    """
    Retrieve detected communities from the graph.
    """
    try:
        with driver.session() as session:
            query = """
                MATCH (cs:CommunitySummary)
                RETURN cs.doc_id AS doc_id, cs.community AS community, cs.summary AS summary
            """
            communities = run_cypher_query(driver, query)
        return {"communities": communities}
    except Exception as e:
        logger.error(f"Error retrieving communities: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving communities: {e}")

@router.get("/community_summaries")
async def community_summaries(doc_id: Optional[str] = None):
    """
    Retrieve stored community summaries.
    """
    try:
        with driver.session() as session:
            if doc_id:
                query = """
                MATCH (cs:CommunitySummary {doc_id: $doc_id})
                RETURN cs.community AS community, cs.summary AS summary, cs.embedding AS embedding
                """
                results = run_cypher_query(driver, query, {"doc_id": doc_id})
            else:
                query = """
                MATCH (cs:CommunitySummary)
                RETURN cs.community AS community, cs.summary AS summary, cs.embedding AS embedding
                """
                results = run_cypher_query(driver, query)
        return {"community_summaries": results}
    except Exception as e:
        logger.error(f"Error retrieving community summaries: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving community summaries: {e}")
