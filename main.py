import os
import time
import re
from neo4j import GraphDatabase
import logging
from fastapi_offline import FastAPIOffline
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
from dotenv import load_dotenv
import requests
import docx
import PyPDF2
from functools import lru_cache
import string

# Load environment variables
load_dotenv()

# Read sleep duration from .env and default to 0.5 seconds if not provided
SLEEP_DURATION = float(os.getenv("SLEEP_DURATION", "0.5"))

# Set up the embedding API endpoint and OpenAI API configurations
embedding_api_url = os.getenv("EMBEDDING_API_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_STOP = os.getenv("OPENAI_STOP")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))


# Logger Class
class Logger:
    def __init__(self, namespace='AppLogger', log_dir='logs', log_file='app.log'):
        log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
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


# Create a global logger instance
global_logger = Logger("AppLogger").get_logger()


# OpenAI Client Wrapper Class
class OpenAIClient:
    def __init__(self):
        self.model = MODEL
        self.stop = OPENAI_STOP
        self.temperature = OPENAI_TEMPERATURE
        self.base_url = OPENAI_BASE_URL
        self.api_key = OPENAI_API_KEY
        self.embedding_api_url = embedding_api_url
        self.logger = Logger("OpenAIClient").get_logger()

    def call_chat_completion(self, messages: List[Dict[str, str]]) -> str:
        # Construct the request payload
        request_payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stop": self.stop
        }
        # Use the base URL directly
        request_url = f"{self.base_url}"
        self.logger.info("Calling OpenAI chat completion API...")
        self.logger.debug(f"Request URL: {request_url}")
        self.logger.debug(f"Request payload: {request_payload}")
        
        try:
            response = requests.post(
                request_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=request_payload,
                timeout=1200  # Timeout in seconds
            )
            response.raise_for_status()
            self.logger.debug(f"Chat completion response: {response.text}")
    
            output = response.json()["choices"][0]["message"]["content"]
            # Remove any lingering stop sequences
            output = output.replace("<|eot_id|>", "")
            
            # Sleep for the duration specified in .env
            time.sleep(SLEEP_DURATION)
    
            self.logger.info("Completed OpenAI chat completion API call.")
            return output
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Error response from OpenAI chat completion API: {e.response.text}")
            self.logger.error(f"Error in OpenAI chat completion request: {e}")
            raise HTTPException(status_code=500, detail="Error in OpenAI chat completion request")



    def get_embeddings(self, text: str) -> List[float]:
        """
        Get embeddings for a given text using the embedding API.
        """
        self.logger.info("Requesting embeddings for text...")
        try:
            response = requests.post(
                f"{self.embedding_api_url}",
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
            self.logger.error(f"Error in OpenAI embedding request: {e}")
            raise HTTPException(status_code=500, detail="Error in OpenAI embedding request")


# Document Processor Class
class DocumentProcessor:
    logger = Logger("DocumentProcessor").get_logger()

    def __init__(self, client: OpenAIClient):
        self.client = client
        
    def clean_text(self, text: str) -> str:
        """
        Clean up the input text by trimming whitespace and normalizing spaces.
        """
        # Strip leading/trailing whitespace
        cleaned = text.strip()
        # Replace multiple whitespace characters (including newlines) with a single space
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # Remove non-printable characters
        cleaned = ''.join(filter(lambda x: x in string.printable, cleaned))
        return cleaned

    def split_documents(self, documents: List[str], chunk_size: int = 600, overlap_size: int = 100) -> List[str]:
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
            self.logger.debug(f"Extracting elements and relationship strength from chunk {index + 1}")
            try:
                response = self.client.call_chat_completion([
                    {"role": "system",
                     "content": "Extract entities, relationships, and their strength from the following text. "
                                "Use common terms such as 'related to', 'depends on', 'influences', etc., "
                                "for relationships, and estimate a strength between 0.0 (very weak) and 1.0 ("
                                "very strong). Format: Parsed relationship: Entity1 -> Relationship -> Entity2 "
                                "[strength: X.X]. Do not include any other text in your response. Use this "
                                "exact format: Parsed relationship: Entity1 -> Relationship -> Entity2 ["
                                "strength: X.X]."},
                    {"role": "user", "content": chunk}
                ])
                self.logger.info(f"Completed extraction for chunk {index + 1}")
                elements.append(response)
            except Exception as e:
                self.logger.error(f"An unexpected error occurred during extraction: {str(e)}")
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
                     "content": "Summarize the following entities and relationships in a structured "
                                "format. Use common terms such as 'related to', 'depends on', "
                                "'influences', etc., for relationships. Use '->' to represent "
                                "relationships after the 'Relationships:' word."},
                    {"role": "user", "content": element}
                ])
                self.logger.info(f"Completed summarization for element {index + 1}")
                summaries.append(summary)
            except Exception as e:
                self.logger.error(f"An unexpected error occurred during summarization: {str(e)}")
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
                self.logger.debug(embedding)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.error(f"Error retrieving embeddings: {e}")
                raise HTTPException(status_code=500, detail="Error retrieving embeddings")
        self.logger.info("Finished generation of embeddings.")
        return embeddings


# Graph Database Connection Class
class GraphDatabaseConnection:
    def __init__(self, uri: str, user: str, password: str) -> None:
        if not uri or not user or not password:
            raise ValueError("URI, user, and password must be provided to initialize the DatabaseConnection.")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def get_session(self):
        """
        Create a context-managed Neo4j session.
        """
        return self.driver.session()

    def clear_database(self) -> None:
        """
        Clear all nodes and relationships from the database.
        """
        with self.get_session() as session:
            session.run("MATCH (n) DETACH DELETE n")


# Graph Manager Class with Graph Data Science (GDS) Integration
class GraphManager:
    logger = Logger('GraphManager').get_logger()

    def __init__(self, db_connection, clear_on_startup: bool = False):
        """
        Initialize graph manager and optionally clear the database.
        """
        self.db_connection = db_connection
        if clear_on_startup:
            self.logger.info("Clearing the graph database on startup...")
            self.db_connection.clear_database()
            
    def vector_search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a vector search on Chunk nodes by computing cosine similarity
        between each chunk's embedding and the provided query embedding.
        """
        self.logger.info("Starting vector search using cosine similarity...")
        with self.db_connection.get_session() as session:
            query = """
            MATCH (c:Chunk)
            WITH c, gds.similarity.cosine(c.embedding, $query_embedding) AS similarity
            RETURN c.text AS text, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """
            result = session.run(query, query_embedding=query_embedding, limit=limit).data()
        self.logger.info("Vector search completed.")
        return result

    def build_graph(self, chunks: List[str], embeddings: List[List[float]], summaries: List[str]) -> None:
        self.logger.info("Starting graph construction...")
        start_time = time.time()
        if self.db_connection is None:
            self.logger.error("Graph database connection is not available.")
            return

        with self.db_connection.get_session() as session:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                self.logger.debug(f"Creating node for chunk {i}")
                session.run("MERGE (c:Chunk {id: $id}) SET c.text = $text, c.embedding = $embedding", id=i, text=chunk, embedding=embedding)

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
                        self.logger.debug(f"Creating relationship: {source} -> {relation_name} -> {target} with weight {weight}")
                        session.run(
                            """
                            MERGE (a:Entity {name: $source})
                            MERGE (b:Entity {name: $target})
                            MERGE (a)-[r:RELATES_TO {weight: $weight}]->(b)
                            """,
                            source=source,
                            target=target,
                            weight=weight
                        )
        end_time = time.time()
        self.logger.debug(f"Graph built in {end_time - start_time:.2f} seconds")
        self.logger.info("Finished graph construction.")

    def calculate_centrality_measures(self, graph_name="entityGraph") -> Dict[str, List[Dict[str, Any]]]:
        self.logger.info("Starting centrality measures calculation...")
        try:
            self.reproject_graph(graph_name)
            self.logger.debug(f"Running centrality measures on graph: {graph_name}")
            with self.db_connection.get_session() as session:
                degree_centrality_query = """
                CALL gds.degree.stream($graph_name)
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).name AS entityId, score
                ORDER BY score DESC
                LIMIT 10
                """
                degree_centrality_result = session.run(degree_centrality_query, graph_name=graph_name).data()

                betweenness_centrality_query = """
                CALL gds.betweenness.stream($graph_name)
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).name AS entityId, score
                ORDER BY score DESC
                LIMIT 10
                """
                betweenness_centrality_result = session.run(betweenness_centrality_query, graph_name=graph_name).data()

                closeness_centrality_query = """
                CALL gds.closeness.stream($graph_name)
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).name AS entityId, score
                ORDER BY score DESC
                LIMIT 10
                """
                closeness_centrality_result = session.run(closeness_centrality_query, graph_name=graph_name).data()

                centrality_data = {
                    "degree": degree_centrality_result,
                    "betweenness": betweenness_centrality_result,
                    "closeness": closeness_centrality_result
                }
            self.logger.info("Finished centrality measures calculation.")
            return centrality_data
        except Exception as e:
            self.logger.error(f"Error calculating centrality measures: {e}")
            raise HTTPException(status_code=500, detail="Error calculating centrality measures")

    def reproject_graph(self, graph_name="entityGraph") -> None:
        self.logger.info("Starting graph projection for GDS...")
        with self.db_connection.get_session() as session:
            check_query = "CALL gds.graph.exists($graph_name) YIELD exists"
            exists_result = session.run(check_query, graph_name=graph_name).single()["exists"]
            if exists_result:
                session.run("CALL gds.graph.drop($graph_name)", graph_name=graph_name)
            session.run("CALL gds.graph.project($graph_name, ['Entity'], '*')", graph_name=graph_name)
            self.logger.debug(f"Graph projection '{graph_name}' created.")
        self.logger.info("Finished graph projection.")

    def normalize_entity_name(self, name: str) -> str:
        """
        Normalize entity names to lowercase for consistency.
        """
        return name.strip().lower()

    def sanitize_relationship_name(self, name: str) -> str:
        """
        Sanitize relationship names by replacing non-alphanumeric characters with underscores.
        """
        return re.sub(r'\W+', '_', name.strip().lower())


# Query Handler Class
class QueryHandler:
    logger = Logger("QueryHandler").get_logger()

    def __init__(self, graph_manager, client: OpenAIClient):
        self.graph_manager = graph_manager
        self.client = client

    def ask_question(self, query: str) -> str:
        self.logger.info("Starting query processing...")
        try:
            # 1. Calculate centrality measures using GDS.
            centrality_data = self.graph_manager.calculate_centrality_measures()
            centrality_summary = self.summarize_centrality_measures(centrality_data)

            # 2. Get the embedding for the question and perform vector search.
            question_embedding = self.client.get_embeddings(query)
            vector_results = self.graph_manager.vector_search(question_embedding, limit=5)
            vector_context = "\n".join([
                f"Chunk: {record['text']}\nSimilarity: {record['similarity']}"
                for record in vector_results
            ])

            # 3. Combine the information into a prompt.
            prompt = (
                f"Query: {query}\n\n"
                f"Centrality Summary:\n{centrality_summary}\n\n"
                f"Relevant Chunks from vector search:\n{vector_context}"
            )

            # 4. Call the chat completion API with both sets of context.
            response = self.client.call_chat_completion([
                {"role": "system", "content": "Use the provided centrality measures and vector search results to answer the query."},
                {"role": "user", "content": prompt}
            ])
            self.logger.info("Finished query processing.")
            return response
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during query processing: {str(e)}")
            raise HTTPException(status_code=500, detail="Error answering question")

    def summarize_centrality_measures(self, centrality_data: Dict[str, Any]) -> str:
        summary = "### Centrality Measures Summary:\n"
        for key, value in centrality_data.items():
            summary += f"\n#### {key.capitalize()} Centrality:\n"
            for record in value:
                summary += f" - {record['entityId']} with score {record['score']}\n"
        return summary


# Initialize the OpenAIClient
client = OpenAIClient()

# Initialize other components
db_connection = GraphDatabaseConnection(
    uri=os.getenv("DB_URL"),
    user=os.getenv("DB_USERNAME"),
    password=os.getenv("DB_PASSWORD")
)
document_processor = DocumentProcessor(client)
graph_manager = GraphManager(db_connection, clear_on_startup=False)
query_handler = QueryHandler(graph_manager, client)

# FastAPI-offline app configuration
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
                global_logger.info("Reading PDF file")
                reader = PyPDF2.PdfReader(file.file)
                text = " ".join([page.extract_text() for page in reader.pages])
                # Clean the extracted text
                text = document_processor.clean_text(text)
                document_texts.append(text)
                global_logger.info("Finished reading PDF file")
            elif file.filename.endswith(".docx"):
                global_logger.info("Reading DOCX file")
                doc = docx.Document(file.file)
                text = "\n".join([para.text for para in doc.paragraphs])
                # Clean the extracted text
                text = document_processor.clean_text(text)
                document_texts.append(text)
                global_logger.info("Finished reading DOCX file")
            elif file.filename.endswith(".txt"):
                global_logger.info("Reading TXT file")
                text = file.file.read().decode("utf-8")
                # Clean the extracted text
                text = document_processor.clean_text(text)
                document_texts.append(text)
                global_logger.info("Finished reading TXT file")
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.filename}")

        global_logger.debug(f"Documents uploaded and processed in memory in {time.time() - start_time:.2f} seconds")

        global_logger.info("Starting document splitting into chunks...")
        chunks = document_processor.split_documents(document_texts)
        global_logger.info("Finished document splitting.")

        chunks_tuple = tuple(chunks)

        global_logger.info("Starting embedding generation for chunks...")
        embeddings = document_processor.get_embeddings(chunks)
        global_logger.info("Finished embedding generation.")

        global_logger.info("Starting extraction of elements from chunks...")
        elements = document_processor.extract_elements(chunks_tuple)
        global_logger.info("Finished extraction of elements.")

        global_logger.info("Starting summarization of extracted elements...")
        summaries = document_processor.summarize_elements(elements)
        global_logger.info("Finished summarization of elements.")

        global_logger.info("Starting graph building...")
        graph_manager.build_graph(chunks, embeddings, summaries)
        global_logger.info("Finished graph building.")

        global_logger.info("Document processing complete.")
        return JSONResponse(
            content={"message": "Documents processed, embeddings generated, and data saved to the graph."}
        )

    except Exception as e:
        global_logger.error(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")


class QueryRequest(BaseModel):
    query: str


@app.post("/ask_question", summary="Ask a question about the graph data")
async def ask_question(request: QueryRequest) -> JSONResponse:
    global_logger.info("Received query request.")
    response = query_handler.ask_question(request.query)
    global_logger.info("Returning query response.")
    return JSONResponse(content={"answer": response})
