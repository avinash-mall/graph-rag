Graph RAG API Documentation
===========================

Overview
--------

Graph RAG (Graph Retrieval-Augmented Generation) is a FastAPI application that integrates a Neo4j graph database with OpenAI’s language models and an embedding API. The primary purpose of the application is to process and index document contents (such as PDF, DOCX, and TXT files), build a knowledge graph, and provide various search functionalities (global, local, cypher, and drift search) over the indexed data. The application also uses advanced techniques like text chunking, community detection, and dynamic Cypher query generation to answer user queries based solely on the content in the graph.

Table of Contents
-----------------

*   [Overview](#overview)
    
*   [Setup and Requirements](#setup-and-requirements)
    
*   [Configuration](#configuration)
    
*   [Architecture](#architecture)
    
    *   [Environment Configuration](#environment-configuration)
        
    *   [Main Components](#main-components)
        
*   [API Endpoints](#api-endpoints)
    
    *   [Document Upload](#document-upload)
        
    *   [Cypher Search](#cypher-search)
        
    *   [Global Search](#global-search)
        
    *   [Local Search](#local-search)
        
    *   [Drift Search](#drift-search)
        
    *   [Document Management](#document-management)
        
*   [Running the Application](#running-the-application)
    
*   [Additional Notes](#additional-notes)
    

Setup and Requirements
----------------------

### Prerequisites

*   **Python 3.8+**
    
*   **Neo4j** – Ensure Neo4j is installed and running.
    
*   **FastAPI** – For the API server.
    
*   **Uvicorn** – ASGI server to run the FastAPI app.
    
*   **Required Python Libraries:**
    
    *   PyPDF2, docx, json5
        
    *   httpx, dotenv, neo4j, pydantic
        
    *   blingfire (for sentence splitting)
        
    *   Additional libraries for logging, asynchronous programming, and embedding computations.
        

### Installation

1.  Clone the repository containing the script.
    
2.  bashCopypip install -r requirements.txt
    
3.  Configure your environment variables by editing the .env file.
    

Configuration
-------------

The application uses an environment file (.env) to manage configuration settings. Key configurations include:

*   **Application Settings:**
    
    *   APP\_HOST and APP\_PORT – Define the host and port for the API server.
        
*   **Graph and Neo4j:**
    
    *   GRAPH\_NAME – Name of the projected graph.
        
    *   DB\_URL, DB\_USERNAME, DB\_PASSWORD – Connection details for Neo4j.
        
*   **OpenAI & LLM Settings:**
    
    *   OPENAI\_API\_KEY – Your OpenAI API key.
        
    *   OPENAI\_MODEL – Model name (e.g., llama3.2).
        
    *   OPENAI\_BASE\_URL – API base URL.
        
    *   OPENAI\_TEMPERATURE – Sampling temperature for the LLM.
        
    *   OPENAI\_STOP – Tokens to denote the end of responses.
        
*   **Chunking and Global Search Defaults:**
    
    *   Parameters like CHUNK\_SIZE\_GDS, GLOBAL\_SEARCH\_CHUNK\_SIZE, GLOBAL\_SEARCH\_TOP\_N, etc.
        
*   **Embedding API:**
    
    *   EMBEDDING\_API\_URL – Endpoint for generating embeddings.
        
    *   API\_TIMEOUT – Timeout settings.
        
*   **Logging Configuration:**
    
    *   LOG\_DIR, LOG\_FILE, and LOG\_LEVEL – Define logging behavior.
        

Refer to the provided .env file for full details ​.env.

Architecture
------------

### Environment Configuration

The application reads settings from the .env file using python-dotenv. This ensures that sensitive details (like API keys and database credentials) remain configurable and are not hard-coded.

### Main Components

1.  **Helper Functions:**
    
    *   **Async Utilities:** run\_async to run coroutines even if an event loop is already active.
        
    *   **Text Processing:** Functions such as clean\_text, chunk\_text, and resolve\_coreferences\_in\_parts prepare and split document text.
        
    *   **Cypher Query Processing:** Functions like extract\_cypher\_code, fix\_cypher\_query, and validate\_and\_refine\_query help generate and validate Cypher queries.
        
2.  **Graph Managers:**
    
    *   **GraphManager & GraphManagerExtended:** Manage graph construction, community detection, and summarization from the document chunks.
        
    *   **GraphManagerWrapper:** Provides a simplified interface to build the graph, reproject it, and handle community summaries.
        
3.  **LLM and Embedding Clients:**
    
    *   **AsyncOpenAI:** Handles asynchronous calls to the language model for tasks such as query rewriting, summarization, and answer generation.
        
    *   **AsyncEmbeddingAPIClient:** Caches and retrieves embeddings for text segments to support similarity computations.
        
4.  **Search & Query Modules:**
    
    *   **Cypher Search:** Dynamically generates and executes Cypher queries based on user-provided questions.
        
    *   **Global, Local, and Drift Search:** These endpoints implement different strategies to search the graph and aggregate document chunks for a comprehensive answer.
        
5.  **API Endpoints:**The FastAPI endpoints orchestrate the interaction between the client and the backend processing logic.
    

Refer to the source code in graph-rag.py for a detailed look at these components ​graph-rag.

API Endpoints
-------------

### Document Upload

*   **Endpoint:** /upload\_documents
    
*   **Method:** POST
    
*   **Description:** Accepts multiple files (PDF, DOCX, TXT), extracts and cleans text, chunks the content, and builds the graph. It also triggers community summary creation.
    
*   **Request:** Multipart form-data with files.
    
*   **Response:** A success message confirming processing and graph update.
    

### Cypher Search

*   **Endpoint:** /cypher\_search
    
*   **Method:** POST
    
*   **Description:** Processes a question by extracting candidate entities, generating and executing Cypher queries, and then mapping results to a final answer.
    
*   **Request:** JSON payload conforming to the QuestionRequest model.
    
*   **Response:** Final answer, aggregated text from the graph, and details of the executed queries.
    

### Global Search

*   **Endpoint:** /global\_search
    
*   **Method:** POST
    
*   **Description:** Performs a global map-reduce search over community summaries using dynamic key point extraction and aggregation.
    
*   **Request:** JSON payload as defined by GlobalSearchRequest.
    
*   **Response:** A detailed answer generated solely from the provided graph data.
    

### Local Search

*   **Endpoint:** /local\_search
    
*   **Method:** POST
    
*   **Description:** Conducts a search within a specified document (or across documents) by combining conversation history, community summaries, and document text units.
    
*   **Request:** JSON payload as defined by LocalSearchRequest.
    
*   **Response:** Answer generated based on the local context of the document(s).
    

### Drift Search

*   **Endpoint:** /drift\_search
    
*   **Method:** POST
    
*   **Description:** Answers a query by synthesizing community reports and refining follow-up queries, generating a hierarchical response.
    
*   **Request:** JSON payload as defined by DriftSearchRequest.
    
*   **Response:** A drift search answer along with a hierarchy of follow-up queries and answers.
    

### Document Management

*   **List Documents**
    
    *   **Endpoint:** /documents
        
    *   **Method:** GET
        
    *   **Description:** Returns a list of documents currently indexed in the graph.
        
*   **Delete Document**
    
    *   **Endpoint:** /delete\_document
        
    *   **Method:** DELETE
        
    *   **Description:** Deletes a document based on doc\_id or document\_name.
        
*   **Community Data**
    
    *   **Endpoint:** /communities and /community\_summaries
        
    *   **Method:** GET
        
    *   **Description:** Provides community detection results and stored community summaries.
        

### Root Endpoint

*   **Endpoint:** /
    
*   **Method:** GET
    
*   **Description:** Basic health check endpoint confirming that the Graph RAG API is running.
    

Running the Application
-----------------------

To run the application, execute the following command from the project directory:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   uvicorn graph-rag:app --host 0.0.0.0 --port 8000   `

Ensure that the environment variables are set (either via the .env file or your system environment) before starting the server.

Additional Notes
----------------

*   **Error Handling:** The application uses structured logging and proper HTTP exceptions to handle errors during file processing, graph operations, and API calls.
    
*   **LLM Integration:** OpenAI API calls and embedding computations are done asynchronously to optimize throughput.
    
*   **Graph Processing:** The application constructs and reprojects graphs dynamically based on document ingestion, using Neo4j’s GDS library for community detection and graph projection.
    
*   **Scalability:** With careful chunking and asynchronous processing, the application is designed to handle larger documents and multiple concurrent requests.
    

This documentation provides an in-depth overview of the core functionalities and API endpoints of the Graph RAG project. For further details, review the source code in graph-rag.py ​graph-rag and the environment configuration in .env ​.env.
