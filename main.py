"""
Main Application File

This file integrates:
 - Document processing (using the remote Flair API for chunking)
 - Embedding generation via a dedicated embedding API (used for query embeddings)
 - Hybrid search (BM25 + dense vector search) via Elasticsearch (handled by VectorManager)
 - Graph construction and retrieval using Neo4j (GraphManager; no vectors stored)
 - Query handling with OpenAI API responses

Ensure that all necessary environment variables are set in your .env file.
"""

# =============================================================================
# Standard Library Imports
# =============================================================================
import os  # For environment variable access
import time  # For sleep delays
import re  # Regular expressions for text processing
import logging  # Logging of operations
import string  # For filtering printable characters
import uuid  # To generate unique document IDs
import json   # For parsing JSON environment variables
from datetime import datetime  # To handle timestamps
from functools import lru_cache  # To cache results of expensive functions
from typing import List, Dict, Any, Tuple  # For type hints

# =============================================================================
# Third-Party Imports
# =============================================================================
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # Disable SSL warnings
import requests  # For HTTP requests (API calls)
import docx  # For processing DOCX files
import PyPDF2  # For processing PDF files
import blingfire  # For sentence splitting
from dotenv import load_dotenv  # Load environment variables from .env
from fastapi_offline import FastAPIOffline  # FastAPI Offline mode
from fastapi import UploadFile, HTTPException, FastAPI  # FastAPI components
from fastapi.responses import JSONResponse  # For JSON responses
from pydantic import BaseModel  # For request body validation
from neo4j import GraphDatabase, Query  # For Neo4j connectivity

# =============================================================================
# Load Environment Variables
# =============================================================================
load_dotenv()

# =============================================================================
# Run Query Helper Function
# =============================================================================
def run_query(session, query_str: str, **params):
    return session.run(Query(query_str), **params)

timeout = float(os.getenv("API_TIMEOUT", "600"))

# =============================================================================
# Logger Setup
# =============================================================================
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", "app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

class Logger:
    def __init__(self, namespace='AppLogger', log_dir=LOG_DIR, log_file=LOG_FILE):
        log_level = getattr(logging, LOG_LEVEL, logging.INFO)
        self.logger = logging.getLogger(namespace)
        self.logger.setLevel(log_level)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not self.logger.hasHandlers():
            fh = logging.FileHandler(os.path.join(log_dir, log_file))
            fh.setLevel(log_level)
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(fmt)
            ch.setFormatter(fmt)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def get_logger(self) -> logging.Logger:
        return self.logger

global_logger = Logger("AppLogger").get_logger()

# =============================================================================
# Remote API Functions for Flair/Blingfire
# =============================================================================
FLAIR_API_URL = os.getenv('FLAIR_API_URL', 'http://localhost:8001')
def remote_extract_flair_entities(text: str) -> List[Dict[str, str]]:
    if not text.strip():
        global_logger.warning("Empty text passed to remote_extract_flair_entities")
        return []
    try:
        resp = requests.post(
            f"{os.getenv('FLAIR_API_URL', 'http://localhost:8001')}/extract_flair_entities",
            json={"text": text},
            verify=False,
            timeout=timeout
        )
        resp.raise_for_status()
        return resp.json().get("entities", [])
    except Exception as e:
        global_logger.error(f"Error calling remote extract_flair_entities: {e}")
        raise

def remote_chunk_text(text: str, max_chunk_size: int) -> List[str]:
    try:
        resp = requests.post(
            f"{FLAIR_API_URL}/chunk_text",
            json={"text": text, "max_chunk_size": max_chunk_size},
            verify=False,
            timeout=timeout
        )
        resp.raise_for_status()
        return resp.json().get("chunks", [])
    except Exception as e:
        global_logger.error(f"Error calling remote chunk_text: {e}")
        raise

# =============================================================================
# Embedding API Client
# =============================================================================
class EmbeddingAPIClient:
    def __init__(self):
        self.embedding_api_url = os.getenv("EMBEDDING_API_URL", "http://localhost:8081/embedding")
        self.logger = Logger("EmbeddingAPIClient").get_logger()

    def get_embedding(self, text: str) -> List[float]:
        self.logger.info("Requesting embedding via API for text: %s", text[:50])
        try:
            response = requests.post(
                self.embedding_api_url,
                json={"input": text, "model": "mxbai-embed-large"},
                timeout=timeout
            )
            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            if not embedding:
                raise ValueError("Empty embedding")
            self.logger.debug("Received query embedding of length: %d", len(embedding))
            return embedding
        except Exception as e:
            self.logger.error(f"Embedding API error: {e}")
            raise HTTPException(status_code=500, detail="Error in embedding API request")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self.get_embedding(text) for text in texts]

embedding_client = EmbeddingAPIClient()

# =============================================================================
# Vector Manager (Elasticsearch-based vector operations)
# =============================================================================
import warnings
from elasticsearch import Elasticsearch, exceptions, ElasticsearchWarning
from scipy.spatial.distance import cosine

warnings.filterwarnings("ignore", category=ElasticsearchWarning)

ES_HOST_URL = os.getenv('ES_HOST_URL')
ES_USERNAME = os.getenv('ES_USERNAME')
ES_PASSWORD = os.getenv('ES_PASSWORD')
ES_INDEX_NAME = os.getenv('ES_INDEX_NAME')
_es = Elasticsearch(
    hosts=[ES_HOST_URL],
    basic_auth=(ES_USERNAME, ES_PASSWORD),
    verify_certs=False
)

NUM_RESULTS = int(os.getenv("NUM_RESULTS", 10))
NUM_CANDIDATES = int(os.getenv("NUM_CANDIDATES", 100))
MIN_SCORE = float(os.getenv("MIN_SCORE", 1.78))

def format_timestamp(iso_timestamp: str) -> str:
    dt = datetime.fromisoformat(iso_timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

class VectorManager:
    def __init__(self, es_client: Elasticsearch, index_name: str):
        self.es = es_client
        self.index_name = index_name
        self.logger = Logger("VectorManager").get_logger()
        self.ensure_index_exists()

    def ensure_index_exists(self):
        if not self.es.indices.exists(index=self.index_name):
            self.logger.info("Index %s does not exist. Creating it...", self.index_name)
            default_dims = int(os.getenv("VECTOR_DIMENSIONS", "1024"))
            similarity = os.getenv("VECTOR_SIMILARITY_FUNCTION", "cosine")
            mappings = {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": default_dims,
                        "index": True,
                        "similarity": similarity
                    },
                    "document_name": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "text": {"type": "text"}
                }
            }
            self.es.indices.create(index=self.index_name, body={"mappings": mappings})
            self.logger.info("Index %s created.", self.index_name)
        else:
            self.logger.info("Index %s exists.", self.index_name)

    def hybrid_search(self, query: str, alpha: float = 0.5, num_results: int = 10) -> List[Dict[str, Any]]:
        self.logger.info("Starting hybrid search for query: %s", query)
        query_embedding = embedding_client.get_embedding(query)
        self.logger.debug("Obtained query embedding of length: %d", len(query_embedding))
        bm25_query = {"query": {"match": {"text": query}}, "size": num_results}
        self.logger.info("Performing BM25 search in Elasticsearch...")
        bm25_response = self.es.search(index=self.index_name, body=bm25_query)
        bm25_results = bm25_response["hits"]["hits"]
        self.logger.debug("BM25 search returned %d hits", len(bm25_results))
        knn_query = {
            "query_vector": query_embedding,
            "field": "embedding",
            "k": num_results,
            "num_candidates": NUM_CANDIDATES
        }
        self.logger.info("Performing KNN search in Elasticsearch...")
        knn_response = self.es.search(index=self.index_name, knn=knn_query)
        knn_results = knn_response["hits"]["hits"]
        self.logger.debug("KNN search returned %d hits", len(knn_results))
        hybrid_results = {}
        for hit in bm25_results:
            doc_id = hit["_id"]
            bm25_score = hit["_score"]
            source = hit["_source"]
            document_name = source.get("document_name", "Unknown Document")
            timestamp = source.get("timestamp")
            formatted_timestamp = format_timestamp(timestamp) if timestamp else "Unknown"
            hybrid_results[doc_id] = {
                "bm25_score": bm25_score,
                "neural_score": 0,
                "text": source["text"],
                "document_name": document_name,
                "timestamp": formatted_timestamp
            }
        for hit in knn_results:
            doc_id = hit["_id"]
            source = hit["_source"]
            neural_score = 1 - cosine(query_embedding, source["embedding"])
            document_name = source.get("document_name", "Unknown Document")
            timestamp = source.get("timestamp")
            formatted_timestamp = format_timestamp(timestamp) if timestamp else "Unknown"
            if doc_id in hybrid_results:
                hybrid_results[doc_id]["neural_score"] = neural_score
            else:
                hybrid_results[doc_id] = {
                    "bm25_score": 0,
                    "neural_score": neural_score,
                    "text": source["text"],
                    "document_name": document_name,
                    "timestamp": formatted_timestamp
                }
        min_bm25 = min([r["bm25_score"] for r in hybrid_results.values()], default=0)
        max_bm25 = max([r["bm25_score"] for r in hybrid_results.values()], default=1)
        min_neural = min([r["neural_score"] for r in hybrid_results.values()], default=0)
        max_neural = max([r["neural_score"] for r in hybrid_results.values()], default=1)
        for doc_id in hybrid_results:
            hybrid_results[doc_id]["bm25_score"] = (hybrid_results[doc_id]["bm25_score"] - min_bm25) / (max_bm25 - min_bm25 + 1e-5)
            hybrid_results[doc_id]["neural_score"] = (hybrid_results[doc_id]["neural_score"] - min_neural) / (max_neural - min_neural + 1e-5)
            hybrid_results[doc_id]["final_score"] = alpha * hybrid_results[doc_id]["bm25_score"] + (1 - alpha) * hybrid_results[doc_id]["neural_score"]
        sorted_results = sorted(hybrid_results.items(), key=lambda x: x[1]["final_score"], reverse=True)
        self.logger.info("Hybrid search completed with %d results", len(sorted_results))
        return [{"doc_id": doc[0], **doc[1]} for doc in sorted_results[:num_results]]

    def list_documents(self) -> List[Dict[str, Any]]:
        self.logger.info("Listing documents from Elasticsearch, index: %s", self.index_name)
        self.ensure_index_exists()
        try:
            response = self.es.search(index=self.index_name, body={"query": {"match_all": {}}, "size": 10000})
            hits = response['hits']['hits']
            self.logger.debug("Elasticsearch search returned %d hits", len(hits))
            docs = {}
            for hit in hits:
                source = hit["_source"]
                doc_id = source.get("doc_id")
                if not doc_id:
                    continue
                if doc_id not in docs:
                    docs[doc_id] = {
                        "doc_id": doc_id,
                        "document_name": source.get("document_name", "Unknown Document"),
                        "timestamp": source.get("timestamp", "Unknown"),
                        "num_chunks_es": 1
                    }
                else:
                    docs[doc_id]["num_chunks_es"] += 1
            self.logger.info("Found %d unique documents in Elasticsearch", len(docs))
            return list(docs.values())
        except Exception as e:
            self.logger.error("Error listing documents from Elasticsearch: %s", e)
            return []

    def delete_documents(self, doc_id: str) -> Dict[str, Any]:
        self.logger.info("Deleting documents from Elasticsearch for doc_id: %s", doc_id)
        try:
            query = {"query": {"term": {"doc_id.keyword": doc_id}}}
            self.logger.debug("Delete query: %s", query)
            response = self.es.delete_by_query(index=self.index_name, body=query, refresh=True)
            self.logger.info("Elasticsearch delete response: %s", response)
            return response
        except Exception as e:
            self.logger.error("Error deleting documents from Elasticsearch for doc_id %s: %s", doc_id, e)
            raise HTTPException(status_code=500, detail=f"Error deleting documents from Elasticsearch for doc_id {doc_id}")

    def index_documents(self, chunks: List[str], embeddings: List[List[float]], metadata_list: List[Dict[str, Any]]) -> None:
        self.logger.info("Indexing %d documents into Elasticsearch...", len(chunks))
        for i, (chunk, emb, meta) in enumerate(zip(chunks, embeddings, metadata_list)):
            doc = {
                "text": chunk,
                "embedding": emb,
                "doc_id": meta["doc_id"],
                "document_name": meta["document_name"],
                "timestamp": meta["timestamp"]
            }
            try:
                doc_id = f"{meta['doc_id']}_{i}"
                resp = self.es.index(index=self.index_name, id=doc_id, body=doc)
                self.logger.debug("Indexed document id: %s, response: %s", doc_id, resp)
            except Exception as e:
                self.logger.error("Error indexing document id: %s, error: %s", doc_id, e)
        self.logger.info("Indexing complete.")

vector_manager = VectorManager(_es, ES_INDEX_NAME)

# =============================================================================
# Document Processor
# =============================================================================
class DocumentProcessor:
    logger = Logger("DocumentProcessor").get_logger()

    def __init__(self, client: 'OpenAIClient'):
        self.client = client

    def clean_text(self, text: str) -> str:
        return ''.join(filter(lambda x: x in string.printable, text.strip()))

    def split_documents(self, documents: List[str], chunk_size: int) -> List[str]:
        self.logger.info("Chunking documents via remote API with chunk size: %d", chunk_size)
        chunks = []
        for doc in documents:
            c = remote_chunk_text(doc, chunk_size)
            self.logger.debug("Document split into %d chunks.", len(c))
            chunks.extend(c)
        self.logger.info("Total chunks produced: %d", len(chunks))
        return chunks

    @lru_cache(maxsize=1024)
    def extract_elements(self, chunks: Tuple[str, ...]) -> List[str]:
        self.logger.info("Extracting elements from chunks...")
        elems = []
        for idx, chunk in enumerate(chunks):
            try:
                resp = self.client.call_chat_completion([
                    {"role": "system", "content": "Extract entities and relationships in the format: Entity1 -> Relationship -> Entity2 [strength: X.X]."},
                    {"role": "user", "content": chunk}
                ])
                elems.append(resp)
                self.logger.debug("Extracted elements from chunk %d", idx + 1)
            except Exception as e:
                self.logger.error("Error processing chunk %d: %s", idx + 1, e)
                raise HTTPException(status_code=500, detail=f"Error processing chunk {idx + 1}")
        return elems

    def summarize_elements(self, elements: List[str]) -> List[str]:
        self.logger.info("Summarizing extracted elements...")
        summaries = []
        for idx, elem in enumerate(elements):
            try:
                summary = self.client.call_chat_completion([
                    {"role": "system", "content": "Summarize the following using '->' for relationships."},
                    {"role": "user", "content": elem}
                ])
                summaries.append(summary)
                self.logger.debug("Summarized element %d", idx + 1)
            except Exception as e:
                self.logger.error("Error summarizing element %d: %s", idx + 1, e)
            # Continue even if one summarization fails.
        return summaries

    def get_embeddings(self, chunks: List[str]) -> List[List[float]]:
        self.logger.info("Generating embeddings for chunks...")
        texts = [f"passage: {chunk}" for chunk in chunks]
        try:
            embeddings = embedding_client.get_embeddings(texts)
            self.logger.debug("Generated embeddings for %d chunks.", len(embeddings))
            return embeddings
        except Exception as e:
            self.logger.error("Embedding error: %s", e)
            raise HTTPException(status_code=500, detail="Error retrieving embeddings")

# =============================================================================
# Neo4j Graph Management (GDS operations without storing vectors)
# =============================================================================
class GraphDatabaseConnection:
    def __init__(self, uri: str, user: str, password: str) -> None:
        if not uri or not user or not password:
            raise ValueError("URI, user, and password are required")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def get_session(self):
        return self.driver.session()

    def clear_database(self) -> None:
        with self.get_session() as session:
            run_query(session, "MATCH (n) DETACH DELETE n")

    def close(self) -> None:
        self.driver.close()

class GraphManager:
    logger = Logger("GraphManager").get_logger()

    def __init__(self, db_connection: GraphDatabaseConnection, clear_on_startup: bool = False):
        self.db_connection = db_connection
        if clear_on_startup:
            self.logger.info("Clearing graph database...")
            self.db_connection.clear_database()
        self.ensure_fulltext_index_exists()

    def ensure_fulltext_index_exists(self) -> None:
        with self.db_connection.get_session() as session:
            res = session.run(Query("SHOW INDEXES YIELD name WHERE name = $index_name RETURN name"),
                              index_name=os.getenv("FULLTEXT_INDEX_NAME", "chunkFulltext")).data()
            if not res:
                self.logger.info("Creating fulltext index: %s", os.getenv("FULLTEXT_INDEX_NAME", "chunkFulltext"))
                run_query(session,
                          f"CREATE FULLTEXT INDEX {os.getenv('FULLTEXT_INDEX_NAME', 'chunkFulltext')} IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text]")
            else:
                self.logger.info("Fulltext index exists.")

    def build_graph(self, chunks: List[str], summaries: List[str], metadata_list: List[Dict[str, Any]]) -> None:
        self.logger.info("Building graph...")
        with self.db_connection.get_session() as session:
            # Create or update Chunk nodes with composite key (chunk index and doc_id)
            for i, (chunk, meta) in enumerate(zip(chunks, metadata_list)):
                query = """
                    MERGE (c:Chunk {id: $cid, doc_id: $doc_id})
                    ON CREATE SET c.text = $text, c.document_name = $document_name, c.timestamp = $timestamp
                    ON MATCH SET c.text = $text, c.document_name = $document_name, c.timestamp = $timestamp
                """
                run_query(session, query, cid=i, doc_id=meta["doc_id"], text=chunk,
                        document_name=meta.get("document_name"), timestamp=meta.get("timestamp"))
            # (Graph building for entities and relationships remains unchanged)
            for i, (chunk, meta) in enumerate(zip(chunks, metadata_list)):
                entities = remote_extract_flair_entities(chunk)
                names = [e["name"].strip().lower() for e in entities]
                for name in names:
                    self._merge_entity(session, name, meta["doc_id"], i)
                if len(names) > 1:
                    for j in range(len(names)):
                        for k in range(j + 1, len(names)):
                            self._merge_cooccurrence(session, names[j], names[k], meta["doc_id"])
            for i, summary in enumerate(summaries):
                doc_id = metadata_list[i]["doc_id"]
                for line in summary.split("\n"):
                    parts = line.split("->")
                    if len(parts) >= 2:
                        source = parts[0].strip().lower()
                        target = parts[-1].strip().lower()
                        weight_match = re.search(r"\[strength:\s*(\d\.\d)\]", line)
                        weight = float(weight_match.group(1)) if weight_match else 1.0
                        rel_query = """
                            MERGE (a:Entity {name: $source, doc_id: $doc_id})
                            MERGE (b:Entity {name: $target, doc_id: $doc_id})
                            MERGE (a)-[r:RELATES_TO {weight: $weight}]->(b)
                        """
                        run_query(session, rel_query, source=source, target=target, weight=weight, doc_id=doc_id)
        self.logger.info("Graph construction complete.")

    def _merge_entity(self, session, name: str, doc_id: str, chunk_id: int):
        query = "MERGE (e:Entity {name: $name, doc_id: $doc_id}) MERGE (e)-[:MENTIONED_IN]->(c:Chunk {id: $cid, doc_id: $doc_id})"
        run_query(session, query, name=name, doc_id=doc_id, cid=chunk_id)

    def _merge_cooccurrence(self, session, name_a: str, name_b: str, doc_id: str):
        query = (
            "MATCH (a:Entity {name: $name_a, doc_id: $doc_id}), "
            "(b:Entity {name: $name_b, doc_id: $doc_id}) "
            "MERGE (a)-[:CO_OCCURS_WITH]->(b)"
        )
        run_query(session, query, name_a=name_a, name_b=name_b, doc_id=doc_id)

    def calculate_centrality_measures(self, graph_name=os.getenv("GRAPH_NAME", "entityGraph"), doc_id: str = None) -> Dict[str, List[Dict[str, Any]]]:
        self.logger.info("Calculating centrality measures...")
        try:
            self.reproject_graph(graph_name, doc_id)
            with self.db_connection.get_session() as session:
                degree = session.run(Query("""
                    CALL gds.degree.stream($graph_name)
                    YIELD nodeId, score
                    RETURN gds.util.asNode(nodeId).name AS entityId, score
                    ORDER BY score DESC LIMIT 10
                """), graph_name=graph_name).data()
                betweenness = session.run(Query("""
                    CALL gds.betweenness.stream($graph_name)
                    YIELD nodeId, score
                    RETURN gds.util.asNode(nodeId).name AS entityId, score
                    ORDER BY score DESC LIMIT 10
                """), graph_name=graph_name).data()
                closeness = session.run(Query("""
                    CALL gds.closeness.stream($graph_name)
                    YIELD nodeId, score
                    RETURN gds.util.asNode(nodeId).name AS entityId, score
                    ORDER BY score DESC LIMIT 10
                """), graph_name=graph_name).data()
                return {"degree": degree, "betweenness": betweenness, "closeness": closeness}
        except Exception as e:
            self.logger.error(f"Centrality calculation error: {e}")
            raise HTTPException(status_code=500, detail="Error calculating centrality measures")

    def reproject_graph(self, graph_name=os.getenv("GRAPH_NAME", "entityGraph"), doc_id: str = None) -> None:
        self.logger.info("Projecting graph for GDS...")
        with self.db_connection.get_session() as session:
            exists = session.run(Query("CALL gds.graph.exists($graph_name) YIELD exists"),
                                 graph_name=graph_name).single()["exists"]
            if exists:
                run_query(session, "CALL gds.graph.drop($graph_name)", graph_name=graph_name)
            if doc_id:
                node_query = f"MATCH (n:Entity) WHERE n.doc_id = '{doc_id}' RETURN id(n) AS id"
                rel_query = f"MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE a.doc_id = '{doc_id}' AND b.doc_id = '{doc_id}' RETURN id(a) AS source, id(b) AS target, r.weight AS weight"
                run_query(session, """
                    CALL gds.graph.project.cypher($graph_name, $nodeQuery, $relQuery)
                """, graph_name=graph_name, nodeQuery=node_query, relQuery=rel_query)
            else:
                run_query(session, "CALL gds.graph.project($graph_name, ['Entity'], '*')", graph_name=graph_name)

            # Debug: Count number of nodes in the projected graph
            node_count_result = session.run(Query("MATCH (n:Entity) RETURN count(n) AS count")).single()
            if node_count_result:
                self.logger.info("Graph projection complete. Entity node count: %d", node_count_result["count"])
            else:
                self.logger.warning("Graph projection complete, but no Entity nodes were found.")

    @staticmethod
    def normalize_entity_name(name: str) -> str:
        return name.strip().lower()

    @staticmethod
    def sanitize_relationship_name(name: str) -> str:
        return re.sub(r'\W+', '_', name.strip().lower())

    def get_entity_context(self, doc_id: str = None) -> str:
        with self.db_connection.get_session() as session:
            if doc_id:
                query = Query("""
                    MATCH (e:Entity {doc_id: $doc_id})-[r:RELATES_TO]->(other:Entity {doc_id: $doc_id})
                    RETURN e.name AS entity, other.name AS related, r.weight AS weight
                    ORDER BY r.weight DESC LIMIT 10
                """)
                result = session.run(query, doc_id=doc_id)
            else:
                query = Query("""
                    MATCH (e:Entity)-[r:RELATES_TO]->(other:Entity)
                    RETURN e.name AS entity, other.name AS related, r.weight AS weight
                    ORDER BY r.weight DESC LIMIT 10
                """)
                result = session.run(query)
            return "\n".join(f"{rec['entity']} -[{rec['weight']}]→ {rec['related']}" for rec in result)

# =============================================================================
# Query Handler (Hybrid Search + Graph Search)
# =============================================================================
class QueryHandler:
    logger = Logger("QueryHandler").get_logger()

    def __init__(self, graph_manager: GraphManager, client: 'OpenAIClient'):
        self.graph_manager = graph_manager
        self.client = client

    def ask_question(self, query: str) -> Dict[str, Any]:
        self.logger.info("Processing query...")
        expanded_query = self.client.call_chat_completion([
            {"role": "system", "content": (
                "You are a professional query expansion agent who expands queries for both Named entity recognition and Vector Search. Rephrase and expand the following question to include additional context and clarity, but do not change its original meaning. Do not write anything else other than the expanded query."
            )},
            {"role": "user", "content": query}
        ])
        self.logger.debug(f"Expanded query: {expanded_query}")
        vector_results = vector_manager.hybrid_search(expanded_query, num_results=NUM_RESULTS)
        top_doc_id = None
        for item in vector_results:
            if item.get("doc_id"):
                # Extract the original document id by splitting on '_' if needed.
                candidate = item["doc_id"].split("_")[0]
                top_doc_id = candidate
                break
        graph_context = ""
        if top_doc_id:
            graph_context = self.graph_manager.get_entity_context(doc_id=top_doc_id)
            if not graph_context.strip():
                self.logger.warning("Graph context for doc_id %s is empty.", top_doc_id)
        else:
            self.logger.debug("No top_doc_id found for graph context.")

        vector_context = "\n---\n".join([item["text"] for item in vector_results])
        answer_vector = self.client.call_chat_completion([
            {"role": "system", "content": "You are a helpful assistant that answers questions using vector search context. Do not use your prior knowledge."},
            {"role": "user", "content": f"Query: {query}\n\nVector Context:\n{vector_context}\n\nAnswer based solely on this vector context."}
        ])

        # Use external Graph Context API for answer_gds
        GRAPH_CONTEXT_API_URL = os.getenv("GRAPH_CONTEXT_API_URL", "http://localhost:8002/get_graph_context")
        filter_by_docid = os.getenv("FILTER_BY_DOCID", "false").lower() == "true"
        payload = {
            "question": query,
            "summary": f"Extract detailed graph context for document {top_doc_id}" if filter_by_docid and top_doc_id else "Extract detailed graph context.",
            "nodes": [],
            "edges": []
        }
        if filter_by_docid and top_doc_id:
            payload["doc_id"] = top_doc_id

        try:
            response = requests.post(GRAPH_CONTEXT_API_URL, json=payload, timeout=timeout)
            response.raise_for_status()
            gds_data = response.json()
            answer_gds = gds_data.get("combined_response", "")
        except Exception as e:
            self.logger.error("Error calling Graph Context API: %s", e)
            answer_gds = f"Error retrieving graph answer: {e}"

        # Retrieve the sub-question responses (assumed to be in the "answers" field)
        final_answers = gds_data.get("answers", [])

        # Build the combined prompt using answer_vector and the sub-question responses.
        combined_prompt = f"Query: {query}\n\n"
        combined_prompt += f"Answer from Vector Search:\n{answer_vector}\n\n"
        combined_prompt += "Sub-question Responses:\n"
        for item in final_answers:
            combined_prompt += f"Question: {item.get('question', '')}\n"
            combined_prompt += f"Answer: {item.get('llm_response', '')}\n\n"
        combined_prompt += "Using the above context, provide a final comprehensive answer that directly addresses the query."

        answer_combined = self.client.call_chat_completion([
            {"role": "system", "content": "You are a professional assistant that synthesizes information from multiple sources. Provide a detailed and comprehensive answer."},
            {"role": "user", "content": combined_prompt}
        ])
        self.logger.info("Query processing complete.")
        return {
            "answer_vector": answer_vector,
            "answer_gds": answer_gds,
            "answer_combined": answer_combined,
            "vector_results": vector_results
        }

# =============================================================================
# OpenAI Client
# =============================================================================
class OpenAIClient:
    def __init__(self):
        self.model = os.getenv("OPENAI_MODEL")
        self.stop = json.loads(os.getenv("OPENAI_STOP", "[]"))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.logger = Logger("OpenAIClient").get_logger()

    def call_chat_completion(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stop": self.stop
        }
        self.logger.info("Calling OpenAI API...")
        try:
            response = requests.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=timeout,
                verify=False
            )
            response.raise_for_status()
            output = response.json()["choices"][0]["message"]["content"].replace("<|eot_id|>", "")
            time.sleep(float(os.getenv("SLEEP_DURATION", "0.5")))
            self.logger.info("OpenAI API call completed.")
            return output
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise HTTPException(status_code=500, detail="Error in OpenAI API request")

# =============================================================================
# FastAPI Setup & Endpoints
# =============================================================================
app = FastAPIOffline()

@app.get("/")
async def get_status():
    return {"status": "running"}

@app.post("/upload_documents", summary="Upload documents for processing")
async def upload_documents(files: List[UploadFile]) -> JSONResponse:
    start = time.time()
    document_texts = []
    metadata = []
    global_logger.info("Starting document processing...")
    try:
        for file in files:
            global_logger.info("Processing file: %s", file.filename)
            if file.filename.endswith(".pdf"):
                reader = PyPDF2.PdfReader(file.file)
                text = " ".join(page.extract_text() for page in reader.pages)
            elif file.filename.endswith(".docx"):
                doc = docx.Document(file.file)
                text = "\n".join(para.text for para in doc.paragraphs)
            elif file.filename.endswith(".txt"):
                text = file.file.read().decode("utf-8")
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.filename}")
            text = DocumentProcessor(None).clean_text(text)
            doc_id = str(uuid.uuid4())
            metadata.append({
                "doc_id": doc_id,
                "document_name": file.filename,
                "timestamp": datetime.now().isoformat()
            })
            document_texts.append(text)
        global_logger.debug("Document processing time: %.2f seconds", time.time() - start)
        processor = DocumentProcessor(OpenAIClient())
        CHUNK_SIZE_VECTOR = int(os.getenv("CHUNK_SIZE_VECTOR", "512"))
        CHUNK_SIZE_GDS = int(os.getenv("CHUNK_SIZE_GDS", "1024"))
        vector_chunks, gds_chunks = [], []
        vector_metadata, gds_metadata = [], []
        for doc_text, meta in zip(document_texts, metadata):
            vc = remote_chunk_text(doc_text, CHUNK_SIZE_VECTOR)
            gc = remote_chunk_text(doc_text, CHUNK_SIZE_GDS)
            vector_chunks.extend(vc)
            gds_chunks.extend(gc)
            vector_metadata.extend([meta] * len(vc))
            gds_metadata.extend([meta] * len(gc))
        vector_embeddings = processor.get_embeddings(vector_chunks)
        vector_manager.index_documents(vector_chunks, vector_embeddings, vector_metadata)
        elements = processor.extract_elements(tuple(gds_chunks))
        summaries = processor.summarize_elements(elements)
        graph_manager.build_graph(gds_chunks, summaries, gds_metadata)
        global_logger.info("Documents processed, indexed, and graph built.")
        return JSONResponse(content={"message": "Documents processed, indexed, and graph updated."})
    except Exception as e:
        global_logger.error("Document processing error: %s", e)
        raise HTTPException(status_code=500, detail=f"Error processing documents: {e}")

@app.get("/documents", summary="List uploaded documents from both Neo4j and Elasticsearch")
async def list_documents() -> JSONResponse:
    try:
        with db_connection.get_session() as session:
            query = Query("""
                MATCH (c:Chunk)
                RETURN DISTINCT c.doc_id AS doc_id, c.document_name AS document_name, c.timestamp AS timestamp
            """)
            neo4j_results = session.run(query).data()
        neo4j_docs = {doc["doc_id"]: doc for doc in neo4j_results}
        es_docs = vector_manager.list_documents()
        es_docs_dict = {doc["doc_id"]: doc for doc in es_docs}
        combined_docs = {}
        all_ids = set(neo4j_docs.keys()) | set(es_docs_dict.keys())
        for doc_id in all_ids:
            combined_docs[doc_id] = {
                "doc_id": doc_id,
                "document_name": neo4j_docs.get(doc_id, es_docs_dict.get(doc_id, {})).get("document_name", "Unknown"),
                "timestamp": neo4j_docs.get(doc_id, es_docs_dict.get(doc_id, {})).get("timestamp", "Unknown"),
                "in_neo4j": doc_id in neo4j_docs,
                "in_elasticsearch": doc_id in es_docs_dict,
                "num_chunks_es": es_docs_dict.get(doc_id, {}).get("num_chunks_es", 0)
            }
        return JSONResponse(content={"documents": list(combined_docs.values())})
    except Exception as e:
        global_logger.error("Error listing documents: %s", e)
        raise HTTPException(status_code=500, detail="Error listing documents")

@app.delete("/documents/{doc_id}", summary="Delete a document from both Neo4j and Elasticsearch")
async def delete_document(doc_id: str) -> JSONResponse:
    errors = []
    try:
        with db_connection.get_session() as session:
            # Delete all nodes with matching doc_id (chunks, entities, etc.)
            del_query = Query("MATCH (n) WHERE n.doc_id = $doc_id DETACH DELETE n")
            session.run(del_query, doc_id=doc_id)
            neo4j_deleted = True
    except Exception as e:
        neo4j_deleted = False
        errors.append(f"Neo4j deletion error: {e}")
        global_logger.error("Error deleting document %s from Neo4j: %s", doc_id, e)
    try:
        es_response = vector_manager.delete_documents(doc_id)
        es_deleted = True
    except Exception as e:
        es_deleted = False
        errors.append(f"Elasticsearch deletion error: {e}")
        global_logger.error("Error deleting document %s from Elasticsearch: %s", doc_id, e)
    if not neo4j_deleted and not es_deleted:
        raise HTTPException(status_code=500, detail="Error deleting document from both databases")
    else:
        return JSONResponse(content={
            "message": f"Document {doc_id} deleted successfully from Neo4j: {neo4j_deleted}, Elasticsearch: {es_deleted}",
            "errors": errors
        })

class QueryRequest(BaseModel):
    query: str

@app.post("/ask_question", summary="Ask a question using vector and graph search")
async def ask_question(req: QueryRequest) -> JSONResponse:
    global_logger.info("Received query request.")
    answer = query_handler.ask_question(req.query)
    global_logger.info("Returning query response.")
    return JSONResponse(content=answer)

# =============================================================================
# Instantiate Global Objects
# =============================================================================
db_connection = GraphDatabaseConnection(uri=os.getenv("DB_URL"), user=os.getenv("DB_USERNAME"), password=os.getenv("DB_PASSWORD"))
graph_manager = GraphManager(db_connection, clear_on_startup=False)
query_handler = QueryHandler(graph_manager, OpenAIClient())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
