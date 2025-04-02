Graph RAG API Documentation
===========================

Overview
--------

Graph RAG (Graph Retrieval-Augmented Generation) is a FastAPI application that integrates a Neo4j graph database with OpenAI’s language models and an embedding API. Its primary purpose is to process and index document contents (e.g., PDF, DOCX, and TXT files), build a knowledge graph, and provide various search functionalities over the indexed data. The application leverages advanced techniques such as text chunking, community detection, dynamic Cypher query generation, and hierarchical query synthesis to answer user queries based solely on the graph content.

Table of Contents
-----------------

*   [Overview](#overview)
    
*   [Setup and Requirements](#setup-and-requirements)
    
*   [Configuration](#configuration)
    
*   [Architecture](#architecture)
    
    *   [Environment Configuration](#environment-configuration)
        
    *   [Main Components](#main-components)
        
*   [Document Upload Pipeline Logic](#document-upload-pipeline-logic)
    
*   [Search Logic](#search-logic)
    
    *   [Cypher Search](#cypher-search)
        
    *   [Global Search](#global-search)
        
    *   [Local Search](#local-search)
        
    *   [Drift Search](#drift-search)
        
*   [API Endpoints](#api-endpoints)
    
    *   [Document Upload](#document-upload)
        
    *   [Cypher Search](#cypher-search-endpoint)
        
    *   [Global Search](#global-search-endpoint)
        
    *   [Local Search](#local-search-endpoint)
        
    *   [Drift Search](#drift-search-endpoint)
        
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

Document Upload Pipeline Logic
------------------------------

When documents are uploaded via the /upload\_documents endpoint, the following steps occur:

1.  **File Processing and Text Extraction:**
    
    *   The endpoint supports multiple file types (PDF, DOCX, and TXT).
        
    *   For PDFs, text is extracted from each page using PyPDF2.
        
    *   DOCX files are processed using the docx library.
        
    *   TXT files are read directly after decoding.
        
2.  **Text Cleaning and Coreference Resolution:**
    
    *   Extracted text is cleaned using clean\_text to remove non-printable characters.
        
    *   The resolve\_coreferences\_in\_parts function refines the text by replacing ambiguous pronouns with their proper entities, ensuring clarity.
        
3.  **Document Metadata and Chunking:**
    
    *   A unique document ID is generated using UUID.
        
    *   The cleaned text is divided into manageable chunks using the chunk\_text function, which leverages sentence splitting (using Blingfire) and groups sentences up to a maximum character limit.
        
4.  **Graph Construction:**
    
    *   The document chunks and their metadata are passed to the graph manager.
        
    *   For each chunk, an embedding is computed via the embedding API.
        
    *   Each chunk is stored in Neo4j as a Chunk node with properties such as text, document name, timestamp, and its embedding.
        
    *   Relationships are established between Entity nodes (extracted via summarization prompts) and the corresponding Chunk nodes.
        
5.  **Community Summaries:**
    
    *   After building the graph, empty nodes are cleaned.
        
    *   Community detection is performed on the graph using Neo4j’s GDS library.
        
    *   For each community, the related text chunks are aggregated and summarized by calling an LLM, and these summaries are stored along with their embeddings.
        

This pipeline ensures that uploaded documents are transformed into a structured graph representation, making them searchable and analyzable via various endpoints.

Search Logic
------------

The application provides several types of searches to retrieve answers based solely on the indexed document data. Each search method uses different strategies to extract and synthesize information from the graph.

### Cypher Search

**Endpoint:** /cypher\_search

**Logic:**

*   **Entity Extraction:**The LLM extracts candidate entities from the user’s question using the extract\_entity\_keywords function.
    
*   **Entity Similarity:**For each candidate entity, the embedding is computed and compared to the embeddings stored in Neo4j. If the cosine similarity exceeds a threshold (e.g., 0.8), the entity is considered a match.
    
*   **Query Generation:**Depending on the matches:
    
    *   **Primary Strategy:**If matching entities are found, the application generates three Cypher queries per entity. These queries match the entity node and retrieve related Chunk nodes via relationships like RELATES\_TO and MENTIONED\_IN.
        
    *   **Fallback Strategy:**If no candidates match, a fuzzy matching query using apoc.text.jaroWinklerDistance is executed to find entities similar to the query text.
        
*   **Aggregation:**The results from all executed queries are aggregated by extracting unique text chunks.
    
*   **Final Answer Generation:**The aggregated text is used to create a final prompt for the LLM to generate a detailed answer to the original question.
    

### Global Search

**Endpoint:** /global\_search

**Logic:**

*   **Community Summaries Retrieval:**The endpoint retrieves stored community summaries, which represent aggregated content from the graph.
    
*   **Similarity Scoring:**The LLM computes embeddings for the user query and compares them to each community summary’s embedding using cosine similarity. Community reports above a certain relevance threshold are selected.
    
*   **Key Point Extraction:**Selected community reports are further chunked, and for each chunk, an LLM extracts key points (with ratings).
    
*   **Aggregation and Reduction:**Key points from all relevant chunks are aggregated and sorted by their rating. A final prompt is constructed, including these key points and the original query, which is sent to the LLM to synthesize a comprehensive answer.
    

### Local Search

**Endpoint:** /local\_search

**Logic:**

*   **Context Building:**The endpoint combines multiple sources of context:
    
    *   **Conversation History:** If provided, the previous exchanges are included.
        
    *   **Community Summaries:** If a document ID is provided, relevant community summaries (ranked via embedding similarity) are retrieved.
        
    *   **Document Text Units:** A sample of text chunks from the specific document is also fetched.
        
*   **Final Prompt:**These context sources are concatenated to build a comprehensive prompt. The LLM is then asked to generate an answer strictly based on the provided context, ensuring that the answer is localized to the document or subset of documents specified.
    

### Drift Search

**Endpoint:** /drift\_search

**Logic:**

*   **Community Report Selection:**Similar to global search, community summaries are retrieved and selected based on their similarity to the query.
    
*   **Primer Phase:**A primer prompt is sent to the LLM using the selected community reports. This generates an intermediate answer along with follow-up questions aimed at refining the query.
    
*   **Follow-Up Query Refinement:**For each follow-up query:
    
    *   Local context is gathered by querying for related text chunks.
        
    *   An LLM call refines the follow-up query and generates additional follow-up details.
        
*   **Hierarchical Aggregation:**The original query, intermediate answers, and follow-up responses are compiled into a hierarchical structure (drift hierarchy).
    
*   **Final Reduction:**A final reduction prompt that incorporates the entire hierarchy is sent to the LLM. The LLM then produces a final, detailed answer that synthesizes the layered information.
    

API Endpoints
-------------

### Document Upload

*   **Endpoint:** /upload\_documents
    
*   **Method:** POST
    
*   **Description:** Accepts multiple files (PDF, DOCX, TXT), extracts and cleans text, chunks the content, builds the graph, and stores community summaries.
    
*   **Request:** Multipart form-data with files.
    
*   **Response:** A success message confirming processing, graph update, and community summary storage.
    

### Cypher Search Endpoint

*   **Endpoint:** /cypher\_search
    
*   **Method:** POST
    
*   **Description:** Processes a question by extracting candidate entities, generating and executing Cypher queries, and then mapping results to a final answer.
    
*   **Request:** JSON payload conforming to the QuestionRequest model.
    
*   **Response:** Final answer, aggregated text from the graph, and details of the executed queries.
    

### Global Search Endpoint

*   **Endpoint:** /global\_search
    
*   **Method:** POST
    
*   **Description:** Performs a global map-reduce search over community summaries using dynamic key point extraction and aggregation.
    
*   **Request:** JSON payload as defined by GlobalSearchRequest.
    
*   **Response:** A detailed answer generated solely from the provided graph data.
    

### Local Search Endpoint

*   **Endpoint:** /local\_search
    
*   **Method:** POST
    
*   **Description:** Conducts a search within a specified document (or across documents) by combining conversation history, community summaries, and document text units.
    
*   **Request:** JSON payload as defined by LocalSearchRequest.
    
*   **Response:** An answer generated based on the local context of the document(s).
    

### Drift Search Endpoint

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

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyuvicorn graph-rag:app --host 0.0.0.0 --port 8000   `

Ensure that the environment variables are set (either via the .env file or your system environment) before starting the server.

Additional Notes
----------------

*   **Error Handling:**The application uses structured logging and proper HTTP exceptions to handle errors during file processing, graph operations, and API calls.
    
*   **LLM Integration:**OpenAI API calls and embedding computations are done asynchronously to optimize throughput.
    
*   **Graph Processing:**The application constructs and reprojects graphs dynamically based on document ingestion, using Neo4j’s GDS library for community detection and graph projection.
    
*   **Scalability:**With careful chunking and asynchronous processing, the application is designed to handle larger documents and multiple concurrent requests.
    

This comprehensive documentation now includes detailed explanations of both the document upload pipeline and the logic behind each type of search offered by the Graph RAG application. For further details, review the source code in graph-rag.py ​graph-rag and the environment configuration in .env ​.env.
