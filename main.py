import os
import time
import re
import logging
import string
import uuid
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Any, Tuple
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import requests
import docx
import PyPDF2
import blingfire
from dotenv import load_dotenv
from fastapi_offline import FastAPIOffline
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from neo4j import GraphDatabase

# ------------------------------------------------------------------------------
# Configuration & Environment Variables
# ------------------------------------------------------------------------------
load_dotenv()

SLEEP_DURATION = float(os.getenv("SLEEP_DURATION", "0.5"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_STOP = os.getenv("OPENAI_STOP")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))

VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "chunk-embeddings")
VECTOR_DIMENSIONS = int(os.getenv("VECTOR_DIMENSIONS", "1024"))
VECTOR_SIMILARITY_FUNCTION = os.getenv("VECTOR_SIMILARITY_FUNCTION", "cosine")
VECTOR_SEARCH_THRESHOLD = float(os.getenv("VECTOR_SEARCH_THRESHOLD", "0.8"))
FULLTEXT_SEARCH_LIMIT = int(os.getenv("FULLTEXT_SEARCH_LIMIT", "2"))

GRAPH_NAME = os.getenv("GRAPH_NAME", "entityGraph")
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", "app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DB_URL = os.getenv("DB_URL")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
FULLTEXT_INDEX_NAME = os.getenv("FULLTEXT_INDEX_NAME", "chunkFulltext")

# Embedding API Configuration
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:8081/embedding")

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def sentence_based_chunker(text: str, max_chunk_size: int) -> List[str]:
    """
    Splits text into sentences using Blingfire, then greedily packs sentences into chunks.
    Each chunk is filled as close as possible to max_chunk_size without splitting any sentence.
    """
    sentences = [s.strip() for s in blingfire.text_to_sentences(text).splitlines() if s.strip()]
    
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if current_chunk and (len(current_chunk) + len(sentence) + 1) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = sentence if not current_chunk else current_chunk + " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# ------------------------------------------------------------------------------
# Logger Class
# ------------------------------------------------------------------------------
class Logger:
    """
    Custom logger that outputs to both file and console.
    """
    def __init__(self, namespace='AppLogger', log_dir=LOG_DIR, log_file=LOG_FILE):
        log_level = getattr(logging, LOG_LEVEL, logging.INFO)
        self.logger = logging.getLogger(namespace)
        self.logger.setLevel(log_level)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not self.logger.hasHandlers():
            file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
            file_handler.setLevel(log_level)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        return self.logger

global_logger = Logger("AppLogger").get_logger()

# ------------------------------------------------------------------------------
# Clients
# ------------------------------------------------------------------------------
class EmbeddingAPIClient:
    """
    Client to generate text embeddings using an external embedding API.
    """
    def __init__(self):
        self.embedding_api_url = EMBEDDING_API_URL
        self.logger = Logger("EmbeddingAPIClient").get_logger()

    def get_embedding(self, text: str) -> List[float]:
        self.logger.info("Requesting embeddings for text...")
        try:
            response = requests.post(
                self.embedding_api_url,
                json={"input": text, "model": "mxbai-embed-large"},
                timeout=30
            )
            response.raise_for_status()
            full_response = response.json()
            self.logger.debug(f"Embedding API response: {full_response}")
            embedding = full_response.get("embedding", [])
            if not embedding:
                raise ValueError("Received empty embedding from API")
            self.logger.info("Received embeddings successfully.")
            return embedding
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error in embedding API request: {e}")
            raise HTTPException(status_code=500, detail="Error in embedding API request")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            emb = self.get_embedding(text)
            embeddings.append(emb)
        return embeddings

embedding_client = EmbeddingAPIClient()

class OpenAIClient:
    """
    Client for interacting with the OpenAI chat completion API.
    """
    def __init__(self):
        self.model = MODEL
        self.stop = OPENAI_STOP
        self.temperature = OPENAI_TEMPERATURE
        self.base_url = OPENAI_BASE_URL
        self.api_key = OPENAI_API_KEY
        self.logger = Logger("OpenAIClient").get_logger()

    def call_chat_completion(self, messages: List[Dict[str, str]]) -> str:
        request_payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stop": self.stop
        }
        self.logger.info("Calling OpenAI chat completion API...")
        self.logger.debug(f"Request URL: {self.base_url}")
        self.logger.debug(f"Request payload: {request_payload}")
        try:
            response = requests.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=request_payload,
                timeout=600,
                verify=False  # Disables SSL certificate verification
            )
            response.raise_for_status()
            output = response.json()["choices"][0]["message"]["content"]
            output = output.replace("<|eot_id|>", "")
            time.sleep(SLEEP_DURATION)
            self.logger.info("Completed OpenAI chat completion API call.")
            return output
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Error response from OpenAI API: {e.response.text}")
            self.logger.error(f"Error in OpenAI API request: {e}")
            raise HTTPException(status_code=500, detail="Error in OpenAI chat completion request")

# ------------------------------------------------------------------------------
# Document Processing
# ------------------------------------------------------------------------------
class DocumentProcessor:
    """
    Processes documents by cleaning text, splitting into chunks using sentence boundaries,
    extracting and summarizing elements, and generating embeddings.
    """
    logger = Logger("DocumentProcessor").get_logger()

    def __init__(self, client: OpenAIClient):
        self.client = client

    def clean_text(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return ''.join(filter(lambda x: x in string.printable, cleaned))

    def split_documents(self, documents: List[str]) -> List[str]:
        self.logger.info("Starting document chunking using sentence_based_chunker...")
        chunks = []
        for document in documents:
            document_chunks = sentence_based_chunker(document, CHUNK_SIZE)
            chunks.extend(document_chunks)
            self.logger.debug(f"Document split into {len(document_chunks)} chunks.")
        self.logger.info(f"Completed chunking; produced {len(chunks)} chunks.")
        return chunks

    @lru_cache(maxsize=1024)
    def extract_elements(self, chunks: Tuple[str, ...]) -> List[str]:
        self.logger.info("Starting extraction of elements from chunks...")
        start_time = time.time()
        elements = []
        for index, chunk in enumerate(chunks):
            self.logger.debug(f"Extracting elements from chunk {index + 1}")
            try:
                response = self.client.call_chat_completion([
                    {"role": "system",
                     "content": (
                         "Extract entities, relationships, and their strength from the following text. "
                         "Use common terms like 'related to', 'depends on', 'influences', etc., and assign a strength between 0.0 and 1.0. "
                         "Format: Parsed relationship: Entity1 -> Relationship -> Entity2 [strength: X.X]. "
                         "Return only this format."
                     )},
                    {"role": "user", "content": chunk}
                ])
                self.logger.info(f"Extraction complete for chunk {index + 1}")
                elements.append(response)
            except Exception as e:
                self.logger.error(f"Extraction error in chunk {index + 1}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error processing chunk {index + 1}")
        self.logger.debug(f"Extraction completed in {time.time() - start_time:.2f} seconds")
        return elements

    def summarize_elements(self, elements: List[str]) -> List[str]:
        self.logger.info("Starting summarization of elements...")
        start_time = time.time()
        summaries = []
        for index, element in enumerate(elements):
            self.logger.debug(f"Summarizing element {index + 1}")
            try:
                summary = self.client.call_chat_completion([
                    {"role": "system",
                     "content": (
                         "Summarize the following entities and relationships in a structured format. "
                         "Use '->' to indicate relationships."
                     )},
                    {"role": "user", "content": element}
                ])
                self.logger.info(f"Summarization complete for element {index + 1}")
                summaries.append(summary)
            except Exception as e:
                self.logger.error(f"Summarization error in element {index + 1}: {str(e)}")
            # Continue processing even if one summarization fails.
        self.logger.debug(f"Summaries created in {time.time() - start_time:.2f} seconds")
        return summaries

    def get_embeddings(self, chunks: List[str]) -> List[List[float]]:
        self.logger.info("Generating embeddings for text chunks...")
        try:
            passage_texts = [f"passage: {chunk}" for chunk in chunks]
            embeddings = embedding_client.get_embeddings(passage_texts)
            self.logger.debug(f"Generated embeddings for {len(embeddings)} chunks.")
            return embeddings
        except Exception as e:
            self.logger.error(f"Embedding generation error: {e}")
            raise HTTPException(status_code=500, detail="Error retrieving embeddings")

# ------------------------------------------------------------------------------
# Graph Database & Graph Management
# ------------------------------------------------------------------------------
class GraphDatabaseConnection:
    """
    Manages the connection to the Neo4j graph database.
    """
    def __init__(self, uri: str, user: str, password: str) -> None:
        if not uri or not user or not password:
            raise ValueError("URI, user, and password are required for the database connection.")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def get_session(self):
        return self.driver.session()

    def clear_database(self) -> None:
        with self.get_session() as session:
            session.run("MATCH (n) DETACH DELETE n")

class GraphManager:
    """
    Manages graph operations including index creation, node/relationship insertion,
    and performing vector and fulltext searches.
    """
    logger = Logger("GraphManager").get_logger()

    def __init__(self, db_connection, clear_on_startup: bool = False):
        self.db_connection = db_connection
        if clear_on_startup:
            self.logger.info("Clearing graph database on startup...")
            self.db_connection.clear_database()
        self.ensure_vector_index_exists()
        self.ensure_fulltext_index_exists()

    def ensure_vector_index_exists(self) -> None:
        with self.db_connection.get_session() as session:
            result = session.run(
                "SHOW INDEXES YIELD name WHERE name = $index_name RETURN name",
                index_name=VECTOR_INDEX_NAME
            ).data()
            if not result:
                self.logger.info(f"Vector index '{VECTOR_INDEX_NAME}' missing. Creating it.")
                session.run(
                    f"""
                    CREATE VECTOR INDEX `{VECTOR_INDEX_NAME}`
                    FOR (n:Chunk) ON (n.embedding)
                    OPTIONS {{indexConfig: {{`vector.dimensions`: {VECTOR_DIMENSIONS}, `vector.similarity_function`: '{VECTOR_SIMILARITY_FUNCTION}'}}}}
                    """
                )
                self.logger.info("Vector index created.")
            else:
                self.logger.info(f"Vector index '{VECTOR_INDEX_NAME}' exists.")

    def ensure_fulltext_index_exists(self) -> None:
        with self.db_connection.get_session() as session:
            result = session.run(
                "SHOW INDEXES YIELD name WHERE name = $index_name RETURN name",
                index_name=FULLTEXT_INDEX_NAME
            ).data()
            if not result:
                self.logger.info(f"Fulltext index '{FULLTEXT_INDEX_NAME}' missing. Creating it.")
                session.run(f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text]")
                self.logger.info("Fulltext index created.")
            else:
                self.logger.info(f"Fulltext index '{FULLTEXT_INDEX_NAME}' exists.")

    def vector_search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        self.logger.info("Performing vector search...")
        with self.db_connection.get_session() as session:
            query = """
            CALL db.index.vector.queryNodes($index_name, $limit, $query_embedding)
            YIELD node AS chunk, score
            WHERE score > $threshold
            RETURN chunk.text AS text, chunk.doc_id AS doc_id, chunk.document_name AS document_name, chunk.timestamp AS timestamp, score
            ORDER BY score DESC LIMIT $limit
            """
            threshold = VECTOR_SEARCH_THRESHOLD
            result = session.run(
                query,
                query_embedding=query_embedding,
                limit=limit,
                threshold=threshold,
                index_name=VECTOR_INDEX_NAME
            ).data()
        for rec in result:
            self.logger.debug(f"Result snippet: {rec.get('text')[:100]}... Score: {rec.get('score')}")
        return result

    def build_graph(self, chunks: List[str], embeddings: List[List[float]], summaries: List[str], metadata_list: List[Dict[str, Any]]) -> None:
        self.logger.info("Starting graph construction...")
        start_time = time.time()
        with self.db_connection.get_session() as session:
            for i, (chunk, embedding, meta) in enumerate(zip(chunks, embeddings, metadata_list)):
                self.logger.debug(f"Creating node for chunk {i}")
                session.run(
                    """
                    MERGE (c:Chunk {id: $id})
                    SET c.text = $text,
                        c.embedding = $embedding,
                        c.doc_id = $doc_id,
                        c.document_name = $document_name,
                        c.timestamp = $timestamp
                    """,
                    id=i,
                    text=chunk,
                    embedding=embedding,
                    doc_id=meta["doc_id"],
                    document_name=meta["document_name"],
                    timestamp=meta["timestamp"]
                )
            for summary in summaries:
                lines = summary.split("\n")
                for line in lines:
                    parts = line.split("->")
                    if len(parts) >= 2:
                        source = self.normalize_entity_name(parts[0].strip())
                        target = self.normalize_entity_name(parts[-1].strip())
                        relationship_part = parts[1].strip()
                        relation_name = self.sanitize_relationship_name(relationship_part.split("[")[0].strip())
                        strength = re.search(r"\[strength:\s*(\d\.\d)\]", relationship_part)
                        weight = float(strength.group(1)) if strength else 1.0
                        self.logger.debug(f"Creating relationship: {source} -> {relation_name} -> {target} (weight: {weight})")
                        session.run(
                            """
                            MERGE (a:Entity {name: $source, doc_id: $doc_id})
                            MERGE (b:Entity {name: $target, doc_id: $doc_id})
                            MERGE (a)-[r:RELATES_TO {weight: $weight}]->(b)
                            """,
                            source=source,
                            target=target,
                            weight=weight,
                            doc_id=meta["doc_id"]
                        )
        self.logger.debug(f"Graph built in {time.time() - start_time:.2f} seconds")
        self.logger.info("Graph construction complete.")

    def reproject_graph(self, graph_name=GRAPH_NAME, doc_id: str = None) -> None:
        self.logger.info("Projecting graph for GDS...")
        with self.db_connection.get_session() as session:
            # Drop any existing projection
            check_query = "CALL gds.graph.exists($graph_name) YIELD exists"
            exists = session.run(check_query, graph_name=graph_name).single()["exists"]
            if exists:
                session.run("CALL gds.graph.drop($graph_name)", graph_name=graph_name)
            if doc_id:
                # Inline the doc_id into the query strings since parameter maps aren't supported here
                node_query = f"MATCH (n:Entity) WHERE n.doc_id = '{doc_id}' RETURN id(n) AS id"
                rel_query = f"MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE a.doc_id = '{doc_id}' AND b.doc_id = '{doc_id}' RETURN id(a) AS source, id(b) AS target, r.weight AS weight"
                session.run(
                    """
                    CALL gds.graph.project.cypher($graph_name, $nodeQuery, $relQuery)
                    """,
                    graph_name=graph_name,
                    nodeQuery=node_query,
                    relQuery=rel_query
                )
                self.logger.debug(f"Graph projection '{graph_name}' created for doc_id {doc_id}.")
            else:
                session.run("CALL gds.graph.project($graph_name, ['Entity'], '*')", graph_name=graph_name)
                self.logger.debug(f"Graph projection '{graph_name}' created without filtering.")
        self.logger.info("Graph projection complete.")



    def calculate_centrality_measures(self, graph_name=GRAPH_NAME, doc_id: str = None) -> Dict[str, List[Dict[str, Any]]]:
        self.logger.info("Calculating centrality measures...")
        try:
            # Pass doc_id to reproject_graph for filtering if provided.
            self.reproject_graph(graph_name, doc_id)
            with self.db_connection.get_session() as session:
                degree_query = """
                CALL gds.degree.stream($graph_name)
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).name AS entityId, score
                ORDER BY score DESC
                LIMIT 10
                """
                degree = session.run(degree_query, graph_name=graph_name).data()

                betweenness_query = """
                CALL gds.betweenness.stream($graph_name)
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).name AS entityId, score
                ORDER BY score DESC
                LIMIT 10
                """
                betweenness = session.run(betweenness_query, graph_name=graph_name).data()

                closeness_query = """
                CALL gds.closeness.stream($graph_name)
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).name AS entityId, score
                ORDER BY score DESC
                LIMIT 10
                """
                closeness = session.run(closeness_query, graph_name=graph_name).data()

                centrality_data = {
                    "degree": degree,
                    "betweenness": betweenness,
                    "closeness": closeness
                }
            return centrality_data
        except Exception as e:
            self.logger.error(f"Centrality calculation error: {e}")
            raise HTTPException(status_code=500, detail="Error calculating centrality measures")

    def normalize_entity_name(self, name: str) -> str:
        return name.strip().lower()

    def sanitize_relationship_name(self, name: str) -> str:
        return re.sub(r'\W+', '_', name.strip().lower())

# ------------------------------------------------------------------------------
# Query Handling
# ------------------------------------------------------------------------------
class QueryHandler:
    """
    Handles user queries by retrieving relevant document snippets from the graph,
    then generating two answers: one based on vector search and one based on graph centrality.
    """
    logger = Logger("QueryHandler").get_logger()

    def __init__(self, graph_manager: GraphManager, client: OpenAIClient):
        self.graph_manager = graph_manager
        self.client = client

    def summarize_centrality_measures(self, centrality_data: Dict[str, Any]) -> str:
        summary = "### Centrality Measures Summary:\n"
        for key, records in centrality_data.items():
            summary += f"\n#### {key.capitalize()} Centrality:\n"
            for rec in records:
                summary += f" - {rec['entityId']} (score: {rec['score']})\n"
        return summary

    def ask_question(self, query: str) -> Dict[str, Any]:
        self.logger.info("Processing query...")
        try:
            # First, perform vector and fulltext searches.
            query_embedding = embedding_client.get_embeddings([f"query: {query}"])[0]
            vector_results = self.graph_manager.vector_search(query_embedding, limit=5)

            fulltext_results = []
            with self.graph_manager.db_connection.get_session() as session:
                cypher_fulltext = """
                    CALL db.index.fulltext.queryNodes($index_name, $query)
                    YIELD node, score
                    RETURN node.text AS text, node.doc_id AS doc_id, node.document_name AS document_name, node.timestamp AS timestamp, score
                    ORDER BY score DESC
                    LIMIT $limit
                """
                result = session.run(cypher_fulltext, parameters={
                    "index_name": FULLTEXT_INDEX_NAME,
                    "query": query,
                    "limit": FULLTEXT_SEARCH_LIMIT
                })
                for rec in result:
                    fulltext_results.append({
                        "text": rec["text"],
                        "doc_id": rec.get("doc_id"),
                        "document_name": rec.get("document_name"),
                        "timestamp": rec.get("timestamp"),
                        "score": rec["score"]
                    })

            combined_results = vector_results + fulltext_results
            combined_results.sort(key=lambda x: x["score"], reverse=True)
            context = "\n---\n".join([item["text"] for item in combined_results[:5]])

            # Use the top result's doc_id (if any) to filter centrality measures.
            top_doc_id = None
            for item in combined_results:
                if item.get("doc_id"):
                    top_doc_id = item["doc_id"]
                    break

            if top_doc_id:
                self.logger.info(f"Filtering centrality measures for doc_id: {top_doc_id}")
            else:
                self.logger.info("No specific doc_id found, using global centrality measures.")

            centrality_data = self.graph_manager.calculate_centrality_measures(doc_id=top_doc_id)
            centrality_summary = self.summarize_centrality_measures(centrality_data)

            prompt_vector = (
                f"Query: {query}\n\n"
                f"Relevant Context:\n{context}\n\n"
                "Answer based solely on the provided context."
            )
            answer_vector = self.client.call_chat_completion([
                {"role": "system", "content": "You are a helpful assistant that answers questions using the provided context."},
                {"role": "user", "content": prompt_vector}
            ])

            prompt_gds = (
                f"Query: {query}\n\n"
                f"Centrality Summary:\n{centrality_summary}\n\n"
                "Answer based solely on the provided centrality data."
            )
            answer_gds = self.client.call_chat_completion([
                {"role": "system", "content": "You are a helpful assistant that answers questions using provided graph centrality data."},
                {"role": "user", "content": prompt_gds}
            ])

            metadata_dict = {}
            for item in combined_results:
                doc_id = item.get("doc_id")
                if doc_id and doc_id not in metadata_dict:
                    metadata_dict[doc_id] = {
                        "doc_id": doc_id,
                        "document_name": item.get("document_name"),
                        "timestamp": item.get("timestamp")
                    }
            metadata_list = list(metadata_dict.values())

            self.logger.info("Query processing complete.")
            return {"answer_vector": answer_vector, "answer_gds": answer_gds, "metadata": metadata_list}
        except Exception as e:
            self.logger.error(f"Query error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error answering question")

# ------------------------------------------------------------------------------
# FastAPI Application Setup & Endpoints
# ------------------------------------------------------------------------------
chat_client = OpenAIClient()
db_connection = GraphDatabaseConnection(uri=DB_URL, user=DB_USERNAME, password=DB_PASSWORD)
document_processor = DocumentProcessor(chat_client)
graph_manager = GraphManager(db_connection, clear_on_startup=False)
query_handler = QueryHandler(graph_manager, chat_client)

app = FastAPIOffline()

@app.get("/")
async def get_status():
    """Return API status."""
    return {"status": "running"}

@app.post("/upload_documents", summary="Upload documents for processing")
async def upload_documents(files: List[UploadFile]) -> JSONResponse:
    """
    Upload and process documents (PDF, DOCX, TXT). The endpoint cleans, splits, embeds,
    extracts, summarizes, and stores document data in the graph.
    """
    start_time = time.time()
    document_texts = []
    document_metadata = []
    try:
        global_logger.info("Starting document processing...")
        for file in files:
            global_logger.info(f"Processing file: {file.filename}")
            if file.filename.endswith(".pdf"):
                reader = PyPDF2.PdfReader(file.file)
                text = " ".join([page.extract_text() for page in reader.pages])
                text = document_processor.clean_text(text)
            elif file.filename.endswith(".docx"):
                doc = docx.Document(file.file)
                text = "\n".join([para.text for para in doc.paragraphs])
                text = document_processor.clean_text(text)
            elif file.filename.endswith(".txt"):
                text = file.file.read().decode("utf-8")
                text = document_processor.clean_text(text)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.filename}")
            
            doc_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            document_texts.append(text)
            document_metadata.append({
                "doc_id": doc_id,
                "document_name": file.filename,
                "timestamp": timestamp
            })
        
        global_logger.debug(f"Document processing time: {time.time() - start_time:.2f} seconds")
        
        all_chunks = []
        all_metadata = []
        for doc_text, meta in zip(document_texts, document_metadata):
            chunks = document_processor.split_documents([doc_text])
            all_chunks.extend(chunks)
            all_metadata.extend([meta] * len(chunks))
        
        chunks_tuple = tuple(all_chunks)
        embeddings = document_processor.get_embeddings(all_chunks)
        elements = document_processor.extract_elements(chunks_tuple)
        summaries = document_processor.summarize_elements(elements)
        graph_manager.build_graph(all_chunks, embeddings, summaries, all_metadata)
        global_logger.info("Document processing completed successfully.")
        return JSONResponse(content={"message": "Documents processed, embeddings generated, and data saved to the graph."})
    except Exception as e:
        global_logger.error(f"Document processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

@app.get("/documents", summary="List all uploaded documents")
async def list_documents() -> JSONResponse:
    """
    List all unique documents that have been processed and stored in the graph.
    """
    try:
        with db_connection.get_session() as session:
            query = """
                MATCH (c:Chunk)
                RETURN DISTINCT c.doc_id AS doc_id, c.document_name AS document_name, c.timestamp AS timestamp
            """
            results = session.run(query).data()
        return JSONResponse(content={"documents": results})
    except Exception as e:
        global_logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Error listing documents")

@app.delete("/documents/{doc_id}", summary="Delete a document")
async def delete_document(doc_id: str) -> JSONResponse:
    """
    Delete all chunks associated with a specific document id from the graph.
    """
    try:
        with db_connection.get_session() as session:
            query_check = "MATCH (c:Chunk {doc_id: $doc_id}) RETURN count(c) as count"
            result = session.run(query_check, doc_id=doc_id).single()
            if result["count"] == 0:
                raise HTTPException(status_code=404, detail="Document not found")
            delete_query = "MATCH (c:Chunk {doc_id: $doc_id}) DETACH DELETE c"
            session.run(delete_query, doc_id=doc_id)
        return JSONResponse(content={"message": f"Document {doc_id} deleted successfully"})
    except HTTPException as he:
        raise he
    except Exception as e:
        global_logger.error(f"Error deleting document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Error deleting document")

class QueryRequest(BaseModel):
    query: str


@app.post("/ask_question", summary="Ask a question about the graph data")
async def ask_question(req: QueryRequest) -> JSONResponse:
    """
    Process a user query by retrieving relevant graph data and generating two separate answers:
    - answer_vector: based on hybrid vector search results.
    - answer_gds: based on graph centrality (GDS) data.
    """
    global_logger.info("Received query request.")
    answer_data = query_handler.ask_question(req.query)
    global_logger.info("Returning query response.")
    return JSONResponse(content=answer_data)
