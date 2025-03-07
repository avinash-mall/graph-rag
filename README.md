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
     docker run -it --rm 
      --publish=7474:7474 --publish=7687:7687 
      --user="$(id -u):$(id -g)" 
      --env NEO4J_AUTH=none 
      --env NEO4J_PLUGINS='["graph-data-science"]' 
      neo4j:latest
   ```
4. **Run Elastic Database**
   ```docker
     docker run -d --name elasticsearch
     -p 9200:9200 -p 9300:9300
     -v $PWD/esdata:/esdata -e "ELASTICSEARCH_DATA_DIR_LIST=/esdata"
     -e "discovery.type=single-node" -e "xpack.security.enabled=false"
     -e ELASTICSEARCH_USERNAME=elastic -e ELASTICSEARCH_PASSWORD=elastic elasticsearch:8.17.3
   ```

5. **(Optionally) Run Embedding and Chat Completion LLMs using LLamafile** 
  ```bash 
    ./Llama-3.2-3B-Instruct.Q6_K.llamafile --server -c 0 --mlock --host 0.0.0.0 --port 8080 --nobrowser
    ./mxbai-embed-large-v1-f16.llamafile --server --nobrowser --embedding --host 0.0.0.0 --port 8081
  ```

6. **Run the uvicorn python api app**
   ```bash
     uvicorn flair_api:app --reload --host 0.0.0.0
   ```
   
7. **Run the uvicorn python app**
   ```bash
     uvicorn main:app --reload --host 0.0.0.0
   ```

8. **Open the swagger-ui docs URL to test**
   ```
     http://<your-ip-address>:8000/docs#
   ```

## Configuration

Create a `.env` file in the project root with the following environment variables:

```env
# OpenAI and Embedding API settings
EMBEDDING_API_URL=https://your-embedding-api-endpoint
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=llama             # or your model of choice
OPENAI_BASE_URL=http://localhost:8080/v1/chat/completions
OPENAI_STOP='["<|end_of_text|>", "<|eot_id|>"]'
OPENAI_TEMPERATURE=0.0
SLEEP_DURATION=0.5

# Neo4j Database Configuration
DB_URL=bolt://localhost:7687
DB_USERNAME=your_db_username
DB_PASSWORD=your_db_password

# Logging
LOG_LEVEL=INFO
```

> **Note:**  
> For a single-instance Neo4j server, use a `bolt://` URI. If you are using a cluster, ensure your URI is correctly configured for routing.

## Running the Application

Start the FastAPI server. For example, if you're using Uvicorn:

```bash
  uvicorn main:app --reload
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

- **POST `/ask_question`**  
  _Graph Query Endpoint_  
  Send a query about the graph (e.g., "Whose friends were Alex and Britt?").  
  Example payload:

  ```json
  {
    "query": "Whose friends were Alex and Britt?"
  }
  ```
## Flowchart

![Alt text](https://raw.githubusercontent.com/avinash-mall/graph-rag/refs/heads/main/flowchart.svg?sanitize=true)
<img src="https://raw.githubusercontent.com/avinash-mall/graph-rag/refs/heads/main/flowchart.svg?sanitize=true">



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
