# GraphRag: Document Processing & Graph Analysis With API

GraphRag is a Python-based application that processes documents, extracts entities and relationships using an OpenAI chat completion API, generates embeddings, and builds a graph in a Neo4j database. It then uses centrality measures computed via Neo4j Graph Data Science (GDS) to answer queries about the graph.

## Features

- **Document Processing:**  
  Supports PDF, DOCX, and TXT files. Extracts text, splits documents into overlapping chunks, and uses chat completions to extract structured entity–relationship information.

- **Embeddings Generation:**  
  Retrieves embeddings for text chunks via an external embedding API.

- **Graph Construction:**  
  Builds a graph in a Neo4j database by creating nodes for text chunks and establishing relationships between entities based on summarizations.

- **Graph Analytics:**  
  Calculates centrality measures (degree, betweenness, and closeness) using Neo4j GDS.

- **Query Handling:**  
  Answers questions by combining the centrality measures with an OpenAI chat completion query.

- **Enhanced Logging:**  
  Provides detailed logging of requests, responses, and internal steps to assist with debugging.

## Requirements

- Python 3.8+
- [FastAPI](https://fastapi.tiangolo.com/)
- [fastapi_offline](https://github.com/dmontagu/fastapi_offline) (or similar)
- [Neo4j Python Driver](https://neo4j.com/developer/python/)
- [Requests](https://docs.python-requests.org/en/latest/)
- [python-dotenv](https://github.com/theskumar/python-dotenv)
- [python-docx](https://python-docx.readthedocs.io/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)

## Installation

1. **Clone the repository:**

   ```bash
     git clone https://github.com/avinash-mall/graph-rag.git
     cd GraphRag
   ```

2. **Install dependencies:**

   ```bash
     pip install -r requirements.txt
   ```

3. **Run Neo4j Database**
   ```docker
    docker run -it -d
   --publish=7474:7474 --publish=7687:7687
   --user="$(id -u):$(id -g)" -e NEO4J_apoc_export_file_enabled=true
   -e NEO4J_apoc_import_file_enabled=true
   -e NEO4J_apoc_import_file_use__neo4j__config=true
   --env NEO4J_AUTH=none --env NEO4J_PLUGINS='["graph-data-science", "apoc"]'
   --name neo4j
   neo4j:latest
   ```

4. **(Optionally) Run Embedding and Chat Completion LLMs using LLamafile** 
  ```bash 
    ollama run llama3.2 #/set num_ctx 32000
    ollama run mxbai-embed-large
  ```

5. **Run the uvicorn python app**
   ```bash
     uvicorn graph-rag:app --reload --host 0.0.0.0
   ```

6. **Open the swagger-ui docs URL to test**
   ```
     http://<your-ip-address>:8000/docs#
   ```


> **Note:**  
> For a single-instance Neo4j server, use a `bolt://` URI. If you are using a cluster, ensure your URI is correctly configured for routing.

## Running the Application

Start the FastAPI server. For example, if you're using Uvicorn:

```bash
  uvicorn graph-rag:app --reload
```

## API Endpoints

- **GET `/`**  
  _Health Check Endpoint_  
  Returns a JSON response confirming the service is running.

- **POST `/upload_documents`**  
  _Document Upload & Processing_  
  Upload one or more files (PDF, DOCX, TXT).  
  Example using \`curl\`:

  ```bash
  curl -X POST "http://localhost:8080/upload_documents" \
       -H "Content-Type: multipart/form-data" \
       -F "files=@/path/to/your/document.pdf"
  ```

- **POST `/global_search`**  
  _Graph Query Endpoint_  
  Send a query about the graph (e.g., "Whose friends were Alex and Britt?").  
  Example payload:

  ```json
  {
    "query": "Whose friends were Alex and Britt?"
  }
  ```

## Logging & Debugging

- **Console & File Logging:**  
  Logs are output to both the console and a file located in the \`logs\` directory.
  
- **Enhanced API Logging:**  
  The application logs the full request payload and response for each call to the OpenAI chat completion API. This helps diagnose issues like timeouts or errors in processing.

## Troubleshooting

- **Timeouts:**  
  If chat completion API calls time out, consider increasing the timeout value in the \`call_chat_completion\` method of the \`OpenAIClient\` class.

- **LLM Issues:**  
  If you encounter LLM unable to process your request, use a larger model as the instruction might be too complex for smaller models.

## Contributing

Contributions are welcome! Please open issues or submit pull requests with improvements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).
`
