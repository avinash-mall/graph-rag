import os
import time
import re
import logging
from neo4j import GraphDatabase
from fastapi_offline import FastAPIOffline
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import requests
import docx
import PyPDF2
from functools import lru_cache
import string
import uuid
from datetime import datetime
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer

# Load configuration from .env file
load_dotenv()

# Application and API configuration constants
SLEEP_DURATION = float(os.getenv("SLEEP_DURATION", "0.5"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_STOP = os.getenv("OPENAI_STOP")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))

# Initialize tokenizer and text splitter for document processing
tokenizer = Tokenizer.from_pretrained("bert-base-multilingual-cased")
splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, CHUNK_SIZE)

# Vector search configuration
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "chunk-embeddings")
VECTOR_DIMENSIONS = int(os.getenv("VECTOR_DIMENSIONS", "1024"))
VECTOR_SIMILARITY_FUNCTION = os.getenv("VECTOR_SIMILARITY_FUNCTION", "cosine")
VECTOR_SEARCH_THRESHOLD = float(os.getenv("VECTOR_SEARCH_THRESHOLD", "0.8"))

# Additional constant to limit fulltext search results
FULLTEXT_SEARCH_LIMIT = int(os.getenv("FULLTEXT_SEARCH_LIMIT", "2"))


# Graph database configuration and logging settings
GRAPH_NAME = os.getenv("GRAPH_NAME", "entityGraph")
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", "app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DB_URL = os.getenv("DB_URL")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
FULLTEXT_INDEX_NAME = os.getenv("FULLTEXT_INDEX_NAME", "chunkFulltext")

# --------------------- Logger Class ---------------------
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

# --------------------- SentenceTransformerClient ---------------------
class SentenceTransformerClient:
    """
    Client to generate text embeddings using SentenceTransformer.
    """
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-large'):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into normalized embeddings."""
        return self.model.encode(texts, normalize_embeddings=True)

# Use environment variable for embedding model if provided
embedding_client = SentenceTransformerClient(os.getenv("EMBEDDING_API_MODEL", "intfloat/multilingual-e5-large"))

# --------------------- OpenAIClient ---------------------
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
        """
        Call the OpenAI API to generate a response based on a list of messages.
        """
        request_payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stop": self.stop
        }
        request_url = self.base_url
        self.logger.info("Calling OpenAI chat completion API...")
        self.logger.debug(f"Request URL: {request_url}")
        self.logger.debug(f"Request payload: {request_payload}")
        try:
            response = requests.post(
                request_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=request_payload,
                timeout=600
            )
            response.raise_for_status()
            self.logger.debug(f"Chat completion response: {response.text}")
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

# --------------------- DocumentProcessor ---------------------
class DocumentProcessor:
    """
    Processes documents by cleaning text, splitting into chunks,
    extracting and summarizing elements, and generating embeddings.
    """
    logger = Logger("DocumentProcessor").get_logger()

    def __init__(self, client: OpenAIClient):
        self.client = client

    def clean_text(self, text: str) -> str:
        """Clean input text by normalizing spaces and filtering non-printable characters."""
        cleaned = text.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return ''.join(filter(lambda x: x in string.printable, cleaned))

    def split_documents(self, documents: List[str]) -> List[str]:
        """
        Split documents into semantic chunks using the pre-configured text splitter.
        """
        self.logger.info("Starting document chunking...")
        chunks = []
        for document in documents:
            document_chunks = splitter.chunks(document)
            chunks.extend(document_chunks)
            self.logger.debug(f"Document split into {len(document_chunks)} chunks.")
        self.logger.info(f"Completed chunking; produced {len(chunks)} chunks.")
        return chunks

    @lru_cache(maxsize=1024)
    def extract_elements(self, chunks: Tuple[str, ...]) -> List[str]:
        """
        Extract entities and relationships from each text chunk using OpenAI.
        Caching is applied to avoid redundant processing.
        """
        self.logger.info("Starting extraction of elements from chunks...")
        start_time = time.time()
        elements = []
        for index, chunk in enumerate(chunks):
            self.logger.debug(f"Extracting elements from chunk {index + 1}")
            try:
                response = self.client.call_chat_completion([
                    {"role": "system",
                     "content": ("Extract entities, relationships, and their strength from the following text. "
                                 "Use common terms like 'related to', 'depends on', 'influences', etc., and assign a strength between 0.0 and 1.0. "
                                 "Format: Parsed relationship: Entity1 -> Relationship -> Entity2 [strength: X.X]. "
                                 "Return only this format.")},
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
        """
        Summarize extracted elements into structured relationships using OpenAI.
        """
        self.logger.info("Starting summarization of elements...")
        start_time = time.time()
        summaries = []
        for index, element in enumerate(elements):
            self.logger.debug(f"Summarizing element {index + 1}")
            try:
                summary = self.client.call_chat_completion([
                    {"role": "system",
                     "content": ("Summarize the following entities and relationships in a structured format. "
                                 "Use '->' to indicate relationships.")},
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
        """
        Generate embeddings for each text chunk using the SentenceTransformer client.
        """
        self.logger.info("Generating embeddings for text chunks...")
        try:
            passage_texts = [f"passage: {chunk}" for chunk in chunks]
            embeddings = embedding_client.get_embeddings(passage_texts)
            self.logger.debug(f"Generated embeddings for {len(embeddings)} chunks.")
            return embeddings
        except Exception as e:
            self.logger.error(f"Embedding generation error: {e}")
            raise HTTPException(status_code=500, detail="Error retrieving embeddings")

# --------------------- GraphDatabaseConnection ---------------------
class GraphDatabaseConnection:
    """
    Manages the connection to the Neo4j graph database.
    """
    def __init__(self, uri: str, user: str, password: str) -> None:
        if not uri or not user or not password:
            raise ValueError("URI, user, and password are required for the database connection.")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        """Close the database connection."""
        self.driver.close()

    def get_session(self):
        """Create a new session for executing database transactions."""
        return self.driver.session()

    def clear_database(self) -> None:
        """Delete all nodes and relationships from the database."""
        with self.get_session() as session:
            session.run("MATCH (n) DETACH DELETE n")

# --------------------- GraphManager ---------------------
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
        """Ensure the vector index exists; create it if missing."""
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
        """Ensure the fulltext index exists; create it if missing."""
        with self.db_connection.get_session() as session:
            result = session.run(
                "SHOW INDEXES YIELD name WHERE name = $index_name RETURN name", 
                index_name=FULLTEXT_INDEX_NAME
            ).data()
            if not result:
                self.logger.info(f"Fulltext index '{FULLTEXT_INDEX_NAME}' missing. Creating it.")
                session.run(f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS FOR (n:Chunk) ON (n.text)")
                self.logger.info("Fulltext index created.")
            else:
                self.logger.info(f"Fulltext index '{FULLTEXT_INDEX_NAME}' exists.")

    def vector_search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a vector search using the provided embedding.
        Returns the top matching nodes with a score above the threshold.
        """
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
        """
        Build the graph by creating nodes for each document chunk and establishing relationships based on summaries.
        """
        self.logger.info("Starting graph construction...")
        start_time = time.time()
        with self.db_connection.get_session() as session:
            # Create nodes for each chunk
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
            # Create relationships between entities as derived from summaries
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
                            MERGE (a:Entity {name: $source})
                            MERGE (b:Entity {name: $target})
                            MERGE (a)-[r:RELATES_TO {weight: $weight}]->(b)
                            """,
                            source=source, target=target, weight=weight
                        )
        self.logger.debug(f"Graph built in {time.time() - start_time:.2f} seconds")
        self.logger.info("Graph construction complete.")

    def calculate_centrality_measures(self, graph_name=GRAPH_NAME) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculate degree, betweenness, and closeness centrality measures using GDS.
        """
        self.logger.info("Calculating centrality measures...")
        try:
            self.reproject_graph(graph_name)
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

    def reproject_graph(self, graph_name=GRAPH_NAME) -> None:
        """
        Reproject the graph for GDS by dropping an existing projection (if present)
        and creating a new one.
        """
        self.logger.info("Projecting graph for GDS...")
        with self.db_connection.get_session() as session:
            check_query = "CALL gds.graph.exists($graph_name) YIELD exists"
            exists = session.run(check_query, graph_name=graph_name).single()["exists"]
            if exists:
                session.run("CALL gds.graph.drop($graph_name)", graph_name=graph_name)
            session.run("CALL gds.graph.project($graph_name, ['Entity'], '*')", graph_name=graph_name)
            self.logger.debug(f"Graph projection '{graph_name}' created.")
        self.logger.info("Graph projection complete.")

    def normalize_entity_name(self, name: str) -> str:
        """Normalize an entity name by stripping whitespace and converting to lowercase."""
        return name.strip().lower()

    def sanitize_relationship_name(self, name: str) -> str:
        """Sanitize relationship names to allow only alphanumeric characters and underscores."""
        return re.sub(r'\W+', '_', name.strip().lower())

# --------------------- QueryHandler ---------------------
class QueryHandler:
    """
    Handles user queries by retrieving relevant document snippets from the graph,
    then generating two answers:
      1. answer_vector: based on hybrid vector search results.
      2. answer_gds: based on graph centrality (GDS) data.
    """
    logger = Logger("QueryHandler").get_logger()

    def __init__(self, graph_manager: GraphManager, client: OpenAIClient):
        self.graph_manager = graph_manager
        self.client = client

    def summarize_centrality_measures(self, centrality_data: Dict[str, Any]) -> str:
        """
        Format centrality measures into a human-readable summary.
        """
        summary = "### Centrality Measures Summary:\n"
        for key, records in centrality_data.items():
            summary += f"\n#### {key.capitalize()} Centrality:\n"
            for rec in records:
                summary += f" - {rec['entityId']} (score: {rec['score']})\n"
        return summary

    def ask_question(self, query: str) -> Dict[str, Any]:
        """
        Process a user query by performing two separate operations:
        1. Generate answer_vector using hybrid vector and fulltext search results.
        2. Generate answer_gds using graph centrality data.
        Both answers are obtained from the OpenAI API using tailored prompts.
        """
        self.logger.info("Processing query...")
        try:
            # Calculate graph centrality measures and prepare a summary.
            centrality_data = self.graph_manager.calculate_centrality_measures()
            centrality_summary = self.summarize_centrality_measures(centrality_data)
    
            # Generate query embedding and perform vector search.
            query_embedding = embedding_client.get_embeddings([f"query: {query}"])[0]
            vector_results = self.graph_manager.vector_search(query_embedding, limit=5)
    
            # Perform fulltext search with a limited number of results.
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
    
            # Combine vector and fulltext results and sort by score.
            combined_results = vector_results + fulltext_results
            combined_results.sort(key=lambda x: x["score"], reverse=True)
            # Use only the top 5 results for context.
            context = "\n---\n".join([item["text"] for item in combined_results[:5]])
    
            # Generate answer_vector using the combined search context.
            prompt_vector = (
                f"Query: {query}\n\n"
                f"Relevant Context:\n{context}\n\n"
                "Answer based solely on the provided context."
            )
            answer_vector = self.client.call_chat_completion([
                {"role": "system", "content": "You are a helpful assistant that answers questions using the provided context."},
                {"role": "user", "content": prompt_vector}
            ])
    
            # Generate answer_gds using the graph centrality summary.
            prompt_gds = (
                f"Query: {query}\n\n"
                f"Centrality Summary:\n{centrality_summary}\n\n"
                "Answer based solely on the provided centrality data."
            )
            answer_gds = self.client.call_chat_completion([
                {"role": "system", "content": "You are a helpful assistant that answers questions using provided graph centrality data."},
                {"role": "user", "content": prompt_gds}
            ])
    
            # Prepare metadata from the combined results.
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


# --------------------- FastAPI Application Setup ---------------------
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
    Upload and process documents (PDF, DOCX, TXT).
    The endpoint cleans, splits, embeds, extracts, summarizes, and stores document data in the graph.
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

class QueryRequest(BaseModel):
    query: str

@app.post("/ask_question", summary="Ask a question about the graph data")
async def ask_question(request: QueryRequest) -> JSONResponse:
    """
    Process a user query by retrieving relevant graph data and generating two separate answers:
      - answer_vector: from the hybrid vector search results.
      - answer_gds: from the graph centrality (GDS) data.
    """
    global_logger.info("Received query request.")
    answer_data = query_handler.ask_question(request.query)
    global_logger.info("Returning query response.")
    return JSONResponse(content=answer_data)
