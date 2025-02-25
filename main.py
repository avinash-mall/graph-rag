import os
import time
import re
import math
from neo4j import GraphDatabase
import logging
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

# Load environment variables from .env file
load_dotenv()

# OpenAI API and Application Configuration
SLEEP_DURATION = float(os.getenv("SLEEP_DURATION", "0.5"))
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_STOP = os.getenv("OPENAI_STOP")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))

# Document Processing Parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
OVERLAP_SIZE = int(os.getenv("OVERLAP_SIZE", "100"))

# Vector Index and Search Parameters
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "chunk-embeddings")
VECTOR_DIMENSIONS = int(os.getenv("VECTOR_DIMENSIONS", "1024"))
VECTOR_SIMILARITY_FUNCTION = os.getenv("VECTOR_SIMILARITY_FUNCTION", "cosine")
VECTOR_SEARCH_THRESHOLD = float(os.getenv("VECTOR_SEARCH_THRESHOLD", "0.8"))

# Graph Projection Parameter (for GDS)
GRAPH_NAME = os.getenv("GRAPH_NAME", "entityGraph")

# Embedding API Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")

# Debug flag (if false, LLM is used to re-rank results)
DEBUG_RERANKING = os.getenv("DEBUG_RERANKING", "false").lower() == "true"

# Logging Configuration
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", "app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Neo4j Database Configuration
DB_URL = os.getenv("DB_URL")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
FULLTEXT_INDEX_NAME = os.getenv("FULLTEXT_INDEX_NAME", "chunkFulltext")


def normalize_vector(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        return [x / norm for x in vec]
    return vec


class Logger:
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


class OpenAIClient:
    def __init__(self):
        self.model = MODEL
        self.stop = OPENAI_STOP
        self.temperature = OPENAI_TEMPERATURE
        self.base_url = OPENAI_BASE_URL
        self.api_key = OPENAI_API_KEY
        self.embedding_api_url = EMBEDDING_API_URL
        self.logger = Logger("OpenAIClient").get_logger()

    def call_chat_completion(self, messages: List[Dict[str, str]]) -> str:
        request_payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stop": self.stop
        }
        request_url = f"{self.base_url}"
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
                self.logger.error(f"Error response from OpenAI chat completion API: {e.response.text}")
            self.logger.error(f"Error in OpenAI chat completion request: {e}")
            raise HTTPException(status_code=500, detail="Error in OpenAI chat completion request")

    def get_embeddings(self, text: str) -> List[float]:
        self.logger.info("Requesting embeddings for text...")
        try:
            response = requests.post(
                f"{self.embedding_api_url}",
                json={"input": text, "model": EMBEDDING_MODEL},
                timeout=30
            )
            response.raise_for_status()
            full_response = response.json()
            self.logger.debug(f"Embedding API response: {full_response}")
            embedding = full_response.get("embedding", [])
            if not embedding:
                raise ValueError("Received empty embedding from API")
            normalized_embedding = normalize_vector(embedding)
            self.logger.debug(f"Normalized embedding: {normalized_embedding}")
            self.logger.info("Received and normalized embeddings successfully.")
            return normalized_embedding
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error in OpenAI embedding request: {e}")
            raise HTTPException(status_code=500, detail="Error in OpenAI embedding request")


class DocumentProcessor:
    logger = Logger("DocumentProcessor").get_logger()

    def __init__(self, client: OpenAIClient):
        self.client = client

    def clean_text(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = ''.join(filter(lambda x: x in string.printable, cleaned))
        return cleaned

    def split_documents(self, documents: List[str], chunk_size: int = CHUNK_SIZE, overlap_size: int = OVERLAP_SIZE) -> List[str]:
        self.logger.info("Starting document splitting into chunks...")
        start_time = time.time()
        chunks = []
        for document in documents:
            for i in range(0, len(document), chunk_size - overlap_size):
                chunk = document[i:i + chunk_size]
                chunks.append(chunk)
        end_time = time.time()
        self.logger.debug(f"Documents split into {len(chunks)} chunks in {end_time - start_time:.2f} seconds")
        self.logger.info("Finished document splitting.")
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
                     "content": "Extract entities, relationships, and their strength from the following text. "
                                "Use common terms such as 'related to', 'depends on', 'influences', etc., and assign a strength between 0.0 and 1.0. "
                                "Format: Parsed relationship: Entity1 -> Relationship -> Entity2 [strength: X.X]. "
                                "Return only this format."},
                    {"role": "user", "content": chunk}
                ])
                self.logger.info(f"Completed extraction for chunk {index + 1}")
                elements.append(response)
            except Exception as e:
                self.logger.error(f"Error during extraction on chunk {index + 1}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error processing chunk {index + 1}")
        end_time = time.time()
        self.logger.debug(f"Elements extracted in {end_time - start_time:.2f} seconds")
        self.logger.info("Finished extraction of elements.")
        return elements

    def summarize_elements(self, elements: List[str]) -> List[str]:
        self.logger.info("Starting summarization of extracted elements...")
        start_time = time.time()
        summaries = []
        for index, element in enumerate(elements):
            self.logger.debug(f"Summarizing element {index + 1}")
            try:
                summary = self.client.call_chat_completion([
                    {"role": "system",
                     "content": "Summarize the following entities and relationships in a structured format. "
                                "Use '->' to indicate relationships."},
                    {"role": "user", "content": element}
                ])
                self.logger.info(f"Completed summarization for element {index + 1}")
                summaries.append(summary)
            except Exception as e:
                self.logger.error(f"Error during summarization on element {index + 1}: {str(e)}")
            # Continue even if one summarization fails.
        end_time = time.time()
        self.logger.debug(f"Summaries created in {end_time - start_time:.2f} seconds")
        self.logger.info("Finished summarization of elements.")
        return summaries

    def get_embeddings(self, chunks: List[str]) -> List[List[float]]:
        self.logger.info("Starting generation of embeddings for chunks...")
        embeddings = []
        for chunk in chunks:
            try:
                embedding = self.client.get_embeddings(chunk)
                self.logger.debug(f"Embedding for chunk: {embedding}")
                embeddings.append(embedding)
            except Exception as e:
                self.logger.error(f"Error retrieving embeddings: {e}")
                raise HTTPException(status_code=500, detail="Error retrieving embeddings")
        self.logger.info("Finished generation of embeddings.")
        return embeddings


class GraphDatabaseConnection:
    def __init__(self, uri: str, user: str, password: str) -> None:
        if not uri or not user or not password:
            raise ValueError("URI, user, and password must be provided for the DatabaseConnection.")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def get_session(self):
        return self.driver.session()

    def clear_database(self) -> None:
        with self.get_session() as session:
            session.run("MATCH (n) DETACH DELETE n")


class GraphManager:
    logger = Logger("GraphManager").get_logger()

    def __init__(self, db_connection, clear_on_startup: bool = False):
        self.db_connection = db_connection
        if clear_on_startup:
            self.logger.info("Clearing the graph database on startup...")
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
                self.logger.info(f"Vector index '{VECTOR_INDEX_NAME}' does not exist. Creating it.")
                session.run(
                    f"""
                    CREATE VECTOR INDEX `{VECTOR_INDEX_NAME}`
                    FOR (n:Chunk) ON (n.embedding)
                    OPTIONS {{indexConfig: {{`vector.dimensions`: {VECTOR_DIMENSIONS}, `vector.similarity_function`: '{VECTOR_SIMILARITY_FUNCTION}'}}}}
                    """
                )
                self.logger.info("Vector index created.")
            else:
                self.logger.info(f"Vector index '{VECTOR_INDEX_NAME}' already exists.")

    def ensure_fulltext_index_exists(self) -> None:
        with self.db_connection.get_session() as session:
            # Use SHOW INDEXES to check for the fulltext index.
            result = session.run(
                "SHOW INDEXES YIELD name WHERE name = $index_name RETURN name", 
                index_name=FULLTEXT_INDEX_NAME
            ).data()
            if not result:
                self.logger.info(f"Fulltext index '{FULLTEXT_INDEX_NAME}' does not exist. Creating it.")
                # Create a fulltext index on Chunk nodes for the 'text' property.
                session.run(f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS FOR (n:Chunk) ON (n.text)")
                self.logger.info("Fulltext index created.")
            else:
                self.logger.info(f"Fulltext index '{FULLTEXT_INDEX_NAME}' already exists.")

    def vector_search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        self.logger.info("Performing vector search using the vector index...")
        with self.db_connection.get_session() as session:
            query = """
            CALL db.index.vector.queryNodes($index_name, $limit, $query_embedding)
            YIELD node AS chunk, score
            WHERE score > $threshold
            RETURN chunk.text AS text, score
            ORDER BY score DESC
            """
            threshold = VECTOR_SEARCH_THRESHOLD
            result = session.run(query, query_embedding=query_embedding, limit=limit, threshold=threshold, index_name=VECTOR_INDEX_NAME).data()
        for rec in result:
            self.logger.debug(f"Vector search result snippet: {rec.get('text')[:100]}... Score: {rec.get('score')}")
        self.logger.info("Vector search completed.")
        return result


    def build_graph(self, chunks: List[str], embeddings: List[List[float]], summaries: List[str]) -> None:
        self.logger.info("Starting graph construction...")
        start_time = time.time()
        with self.db_connection.get_session() as session:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                self.logger.debug(f"Creating node for chunk {i}")
                session.run(
                    "MERGE (c:Chunk {id: $id}) SET c.text = $text, c.embedding = $embedding", 
                    id=i, text=chunk, embedding=embedding
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
                            MERGE (a:Entity {name: $source})
                            MERGE (b:Entity {name: $target})
                            MERGE (a)-[r:RELATES_TO {weight: $weight}]->(b)
                            """,
                            source=source, target=target, weight=weight
                        )
        end_time = time.time()
        self.logger.debug(f"Graph built in {end_time - start_time:.2f} seconds")
        self.logger.info("Finished graph construction.")

    def calculate_centrality_measures(self, graph_name=GRAPH_NAME) -> Dict[str, List[Dict[str, Any]]]:
        self.logger.info("Calculating centrality measures using GDS...")
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
            self.logger.info("Centrality measures calculated successfully.")
            return centrality_data
        except Exception as e:
            self.logger.error(f"Error calculating centrality measures: {e}")
            raise HTTPException(status_code=500, detail="Error calculating centrality measures")

    def reproject_graph(self, graph_name=GRAPH_NAME) -> None:
        self.logger.info("Projecting graph for GDS...")
        with self.db_connection.get_session() as session:
            check_query = "CALL gds.graph.exists($graph_name) YIELD exists"
            exists = session.run(check_query, graph_name=graph_name).single()["exists"]
            if exists:
                session.run("CALL gds.graph.drop($graph_name)", graph_name=graph_name)
            session.run("CALL gds.graph.project($graph_name, ['Entity'], '*')", graph_name=graph_name)
            self.logger.debug(f"Graph projection '{graph_name}' created.")
        self.logger.info("Finished graph projection.")

    def normalize_entity_name(self, name: str) -> str:
        return name.strip().lower()

    def sanitize_relationship_name(self, name: str) -> str:
        return re.sub(r'\W+', '_', name.strip().lower())


class QueryHandler:
    logger = Logger("QueryHandler").get_logger()

    def __init__(self, graph_manager: GraphManager, client: OpenAIClient):
        self.graph_manager = graph_manager
        self.client = client

    def rerank_vector_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates_text = "\n".join([f"{i+1}. {record['text']}" for i, record in enumerate(results)])
        prompt = (
            f"Given the query:\n{query}\n\n"
            f"And these document snippets:\n{candidates_text}\n\n"
            "Return a comma-separated list of the snippet numbers ordered by relevance."
        )
        self.logger.info("Requesting re-ranking from LLM...")
        response = self.client.call_chat_completion([
            {"role": "system", "content": "Re-rank the following document snippets based solely on relevance to the query."},
            {"role": "user", "content": prompt}
        ])
        self.logger.debug(f"Re-ranking response: {response}")
        try:
            ranking_order = [int(num) for num in re.findall(r'\d+', response)]
        except Exception as e:
            self.logger.error(f"Error parsing re-ranking response: {e}")
            ranking_order = list(range(1, len(results) + 1))
        reranked = []
        seen = set()
        for pos in ranking_order:
            idx = pos - 1
            if 0 <= idx < len(results) and idx not in seen:
                reranked.append(results[idx])
                seen.add(idx)
        for i, item in enumerate(results):
            if i not in seen:
                reranked.append(item)
        return reranked

    def summarize_centrality_measures(self, centrality_data: Dict[str, Any]) -> str:
        summary = "### Centrality Measures Summary:\n"
        for key, records in centrality_data.items():
            summary += f"\n#### {key.capitalize()} Centrality:\n"
            for rec in records:
                summary += f" - {rec['entityId']} (score: {rec['score']})\n"
        return summary

    def ask_question(self, query: str) -> str:
        self.logger.info("Processing query...")
        try:
            # Calculate centrality measures using GDS and get a summary.
            centrality_data = self.graph_manager.calculate_centrality_measures()
            centrality_summary = self.summarize_centrality_measures(centrality_data)

            # Retrieve vector search results.
            query_embedding = self.client.get_embeddings(query)
            vector_results = self.graph_manager.vector_search(query_embedding, limit=5)

            # Retrieve fulltext search results using the fulltext index.
            fulltext_results = []
            with self.graph_manager.db_connection.get_session() as session:
                cypher_fulltext = """
                CALL db.index.fulltext.queryNodes($index_name, $query)
                YIELD node, score
                RETURN node.text AS text, score
                ORDER BY score DESC
                LIMIT $limit
                """
                result = session.run(cypher_fulltext, parameters={
                    "index_name": FULLTEXT_INDEX_NAME,
                    "query": query,
                    "limit": 5
                })
                for rec in result:
                    fulltext_results.append({"text": rec["text"], "score": rec["score"]})

            # Combine and (optionally) re-rank the results.
            combined_results = vector_results + fulltext_results
            combined_results.sort(key=lambda x: x["score"], reverse=True)
            if not DEBUG_RERANKING:
                combined_results = self.rerank_vector_results(query, combined_results)

            context = "\n---\n".join([item["text"] for item in combined_results[:5]])
            prompt = (
                f"Query: {query}\n\n"
                f"Centrality Summary:\n{centrality_summary}\n\n"
                f"Relevant Context:\n{context}\n\n"
                "Answer based solely on the provided context."
            )
            response = self.client.call_chat_completion([
                {"role": "system", "content": "You are a helpful assistant that answers questions using provided context and graph data."},
                {"role": "user", "content": prompt}
            ])
            self.logger.info("Query processing completed.")
            return response
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise HTTPException(status_code=500, detail="Error answering question")


# Initialize components
client = OpenAIClient()
db_connection = GraphDatabaseConnection(uri=DB_URL, user=DB_USERNAME, password=DB_PASSWORD)
document_processor = DocumentProcessor(client)
graph_manager = GraphManager(db_connection, clear_on_startup=False)
query_handler = QueryHandler(graph_manager, client)

app = FastAPIOffline()


@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post("/upload_documents", summary="Upload documents for processing")
async def upload_documents(files: List[UploadFile]) -> JSONResponse:
    start_time = time.time()
    document_texts = []
    try:
        global_logger.info("Starting document upload and processing...")
        for file in files:
            global_logger.info(f"Processing file: {file.filename}")
            if file.filename.endswith(".pdf"):
                reader = PyPDF2.PdfReader(file.file)
                text = " ".join([page.extract_text() for page in reader.pages])
                text = document_processor.clean_text(text)
                document_texts.append(text)
            elif file.filename.endswith(".docx"):
                doc = docx.Document(file.file)
                text = "\n".join([para.text for para in doc.paragraphs])
                text = document_processor.clean_text(text)
                document_texts.append(text)
            elif file.filename.endswith(".txt"):
                text = file.file.read().decode("utf-8")
                text = document_processor.clean_text(text)
                document_texts.append(text)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.filename}")
        global_logger.debug(f"Documents processed in {time.time() - start_time:.2f} seconds")
        chunks = document_processor.split_documents(document_texts, CHUNK_SIZE, OVERLAP_SIZE)
        chunks_tuple = tuple(chunks)
        embeddings = document_processor.get_embeddings(chunks)
        elements = document_processor.extract_elements(chunks_tuple)
        summaries = document_processor.summarize_elements(elements)
        graph_manager.build_graph(chunks, embeddings, summaries)
        global_logger.info("Document processing complete.")
        return JSONResponse(content={"message": "Documents processed, embeddings generated, and data saved to the graph."})
    except Exception as e:
        global_logger.error(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")


class QueryRequest(BaseModel):
    query: str


@app.post("/ask_question", summary="Ask a question about the graph data")
async def ask_question(request: QueryRequest) -> JSONResponse:
    global_logger.info("Received query request.")
    answer = query_handler.ask_question(request.query)
    global_logger.info("Returning query response.")
    return JSONResponse(content={"answer": answer})
