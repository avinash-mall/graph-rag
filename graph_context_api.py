import os
import re
import json
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
from neo4j import GraphDatabase
from dotenv import load_dotenv
import requests

# Setup basic logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# ---------------- Neo4j Configuration ----------------
NEO4J_URI = os.getenv("DB_URL", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("DB_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("DB_PASSWORD", "neo4j")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# ---------------- OpenAI Configuration ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_STOP = os.getenv("OPENAI_STOP", "")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

app = FastAPI(
    title="Graph Context API",
    description=("API for generating and executing dynamic Cypher queries using few-shot learning "
                 "based on actual graph stats and schema details."),
    version="1.0.0"
)


# ---------------- Neo4j Query Functions ----------------
def run_cypher_query(query: str, parameters: Dict[str, Any] = {}) -> list:
    logger.debug(f"Running Cypher query: {query} with parameters: {parameters}")
    try:
        with driver.session() as session:
            result = session.run(query, **parameters)
            data = [record.data() for record in result]
            logger.debug(f"Query result: {data}")
            return data
    except Exception as e:
        logger.error(f"Error executing query: {query}. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Neo4j query execution error: {e}")


def get_schema() -> str:
    query = """
    CALL apoc.meta.schema({
      includeLabels: true,
      includeRels: true,
      sample: -1
    })
    YIELD value
    UNWIND keys(value) as label
    WITH label, value[label] as data
    WHERE data.type = "node"
    MATCH (n)
    OPTIONAL MATCH (n)-[r]->(m)
    WITH label, data, 
         collect(DISTINCT {
             type: type(r), 
             direction: "outgoing", 
             targetLabel: head(labels(m))
         }) as outRelations
    OPTIONAL MATCH (n)<-[r]-(m)
    WITH label, data, outRelations,
         collect(DISTINCT {
             type: type(r), 
             direction: "incoming", 
             sourceLabel: head(labels(m))
         }) as inRelations
    RETURN { label: label, properties: data.properties, outgoingRelationships: outRelations, incomingRelationships: inRelations } AS schema
    LIMIT 1
    """
    results = run_cypher_query(query)
    return results[0].get('schema', "No schema found.") if results else "No schema found."


def get_node_counts() -> dict:
    query = """
    MATCH (n)
    RETURN labels(n)[0] AS nodeType, count(*) AS count
    ORDER BY count DESC
    """
    results = run_cypher_query(query)
    return {rec.get("nodeType"): rec.get("count") for rec in results}


def get_edge_counts() -> dict:
    query = """
    MATCH ()-[r]->()
    RETURN type(r) AS edgeType, count(*) AS count
    ORDER BY count DESC
    """
    results = run_cypher_query(query)
    return {rec.get("edgeType"): rec.get("count") for rec in results}


def get_most_connected_nodes() -> list:
    query = """
    MATCH (n)
    OPTIONAL MATCH (n)-[r]-()
    RETURN labels(n)[0] AS nodeType, n.name AS nodeName, COUNT(r) AS connections
    ORDER BY connections DESC
    LIMIT 5
    """
    return run_cypher_query(query)


def get_common_node_properties() -> list:
    query = """
    MATCH (n)
    UNWIND keys(n) AS key
    WITH labels(n)[0] AS nodeType, key
    RETURN nodeType, collect(DISTINCT key) AS properties
    ORDER BY nodeType
    """
    return run_cypher_query(query)


def get_relationship_patterns() -> list:
    query = """
    MATCH (a)-[r]->(b)
    WITH labels(a)[0] + '-[' + type(r) + ']->' + labels(b)[0] AS pattern, count(*) AS count
    RETURN pattern, count
    ORDER BY count DESC
    LIMIT 5
    """
    return run_cypher_query(query)


def generate_graph_structure() -> str:
    query = "CALL apoc.meta.schema() YIELD value RETURN value LIMIT 1"
    results = run_cypher_query(query)
    return str(results[0].get("value")) if results else "No graph structure found."


def get_graph_stats() -> dict:
    return {
        "schema": get_schema(),
        "nodeCounts": get_node_counts(),
        "edgeCounts": get_edge_counts(),
        "mostConnectedNodes": get_most_connected_nodes(),
        "commonNodeProperties": get_common_node_properties(),
        "relationshipPatterns": get_relationship_patterns(),
        "graphStructure": generate_graph_structure()
    }


def reduce_graph_stats(stats: dict) -> dict:
    return {
        "schema": stats.get("schema"),
        "nodeCounts": stats.get("nodeCounts"),
        "edgeCounts": stats.get("edgeCounts"),
        "mostConnectedNodes": stats.get("mostConnectedNodes", [])[:3],
        "commonNodeProperties": [
            {"nodeType": item.get("nodeType"), "properties": item.get("properties", [])[:5]}
            for item in stats.get("commonNodeProperties", [])
        ],
        "relationshipPatterns": stats.get("relationshipPatterns", [])[:3],
        "graphStructure": stats.get("graphStructure")
    }


def get_reduced_graph_stats() -> dict:
    stats = get_graph_stats()
    reduced = reduce_graph_stats(stats)
    logger.debug(f"Reduced graph stats: {json.dumps(reduced, indent=2)}")
    return reduced


# ---------------- Prompt Builder Functions ----------------
def build_cypher_questions_prompt(question: str, summary: str, nodes: list, edges: list) -> str:
    graph_stats = get_reduced_graph_stats()
    available_labels = list(graph_stats.get("nodeCounts", {}).keys())
    available_rels = list(graph_stats.get("edgeCounts", {}).keys())

    prompt = f"""<context> You are an expert in graph databases and Cypher queries. Your task is to generate insightful questions and corresponding Cypher queries strictly using the available node labels and relationship types. Your output must be a JSON array (without markdown formatting) where each element is an object with the keys: "question", "standard_query", "fuzzy_query", "general_query", and "explanation". </context>

<user_question>
  {question}
</user_question>

<graph_schema>
  Available Node Labels: {json.dumps(available_labels)}
  Available Relationship Types: {json.dumps(available_rels)}
  Detailed Schema: {graph_stats.get("schema")}
</graph_schema>

<additional_context>
  Graph Structure: {graph_stats.get("graphStructure")}
</additional_context>

<summary>
  {summary}
</summary>

<query_strategies>
  When formulating queries, consider:
  - Using case-insensitive regex (e.g., =~ '(?i).*term.*')
  - OPTIONAL MATCH for missing relationships or nodes
  - Flexible text matching with CONTAINS or regex
  - Including ORDER BY and LIMIT (typically 5-10 results)
</query_strategies>

<query_structure>
  1. Main MATCH clause(s) for primary entities
  2. WHERE clause with flexible matching (regex/CONTAINS)
  3. OPTIONAL MATCH for related nodes (if needed)
  4. RETURN clause with key data and aggregations
  5. ORDER BY and LIMIT clauses
</query_structure>

<examples>
  Example Neo4j query: MATCH (harris:Entity) RETURN harris;
  Formatted response sample: (:Entity {{name: "deceived miss sutherland by disguising himself as mr. hosmer angel (strength: 0.95)", doc_id: "8df8440f-e295-407a-8690-bcd6bab6769f"}})

  Example Neo4j query: MATCH (harris:Entity)-[r:RELATES_TO]->(other) RETURN harris, r, other;
  Formatted response sample: (:Entity {{name: "here are the summarized relationships using '", doc_id: "8df8440f-e295-407a-8690-bcd6bab6769f"}})
</examples>

<chain_of_thought>
  1. Identify main entities and relationships from the user question.
  2. Start with a broad MATCH clause.
  3. Use regex matching for flexibility.
  4. Include OPTIONAL MATCH for additional context.
  5. Consider alternative query structures for a broader answer.
</chain_of_thought>

Generate 5-7 questions with corresponding Cypher queries (standard, fuzzy, general).
Important 1: The queries must be related in a substantive way to the user's question.
Important 2: The output should be a JSON array where each element is an object with the following keys:
  "question", "standard_query", "fuzzy_query", "general_query", and "explanation".
"""
    return prompt.strip()


def build_llm_answer_prompt(query_context: dict) -> str:
    prompt = f"""Given the data extracted from the graph database:

Question: {query_context.get('question')}

Standard Query Output:
{json.dumps(query_context.get('standard_query_output'), indent=2)}

Fuzzy Query Output:
{json.dumps(query_context.get('fuzzy_query_output'), indent=2)}

General Query Output:
{json.dumps(query_context.get('general_query_output'), indent=2)}

Provide a concise and insightful answer to the user's query based solely on the information above. Do not include any 
extra commentary or markdown formatting."""
    return prompt.strip()


# ---------------- OpenAI Client ----------------
class OpenAIClient:
    def __init__(self):
        self.model = OPENAI_MODEL
        self.base_url = OPENAI_BASE_URL
        self.api_key = OPENAI_API_KEY
        self.temperature = OPENAI_TEMPERATURE

    def call_chat_completion(self, messages: list) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stop": OPENAI_STOP
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=60,
                verify=False
            )
            response.raise_for_status()
            output = response.json()["choices"][0]["message"]["content"]
            logger.debug(f"LLM output:\n{output}")
            return output.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")


# ---------------- LLM Call & JSON Parsing ----------------
def call_llm_for_queries(prompt: str, client: OpenAIClient) -> str:
    messages = [
        {"role": "system", "content": "You are a professional Cypher query generator."},
        {"role": "user", "content": prompt}
    ]
    return client.call_chat_completion(messages)


def parse_llm_output(output: str) -> list:
    """
    Parse the LLM output assuming it is strictly valid JSON.
    The JSON should be an array of objects with keys:
    "question", "standard_query", "fuzzy_query", "general_query", and "explanation".
    """
    try:
        queries = json.loads(output)
        if not isinstance(queries, list):
            raise ValueError("LLM output JSON is not an array")
        logger.debug(f"Parsed queries: {json.dumps(queries, indent=2)}")
        return queries
    except Exception as e:
        logger.error(f"Failed to parse LLM output into JSON: {e}")
        raise HTTPException(status_code=500,
                            detail="Failed to parse LLM output into JSON. Please check the LLM output format.")


# ---------------- Request Model ----------------
class GraphContextRequest(BaseModel):
    question: str
    doc_id: Optional[Union[str, List[str]]] = None
    summary: str = ""
    nodes: list = []
    edges: list = []



def build_combined_answer_prompt(user_question: str, responses: list) -> str:
    prompt = f"User Question: {user_question}\n\n"
    prompt += "The following responses were generated for different aspects of your query:\n"
    for idx, resp in enumerate(responses, start=1):
        prompt += f"{idx}. {resp}\n"
    prompt += ("\nBased on the above responses, please provide a comprehensive and concise final answer "
               "that directly addresses the user's original question.")
    return prompt.strip()

# ---------------- API Endpoint ----------------
@app.post("/get_graph_context",
          summary="Generate and execute dynamic Cypher queries using few-shot learning based on graph stats")
def get_graph_context(request: GraphContextRequest):
    # Prepare human-friendly doc_id filter description
    doc_id_str = ", ".join(request.doc_id) if isinstance(request.doc_id, list) else (request.doc_id or "all documents")
    summary_text = request.summary or f"Extract detailed graph context for {doc_id_str}."

    # Build prompt with enhanced instructions and examples
    prompt = build_cypher_questions_prompt(request.question, summary_text, request.nodes, request.edges)
    logger.debug(f"Generated prompt for LLM:\n{prompt}")

    openai_client = OpenAIClient()
    llm_output = call_llm_for_queries(prompt, openai_client)

    try:
        generated_queries = parse_llm_output(llm_output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing LLM output: {e}")

    # Prepare parameters for Neo4j queries
    params = {
        "doc_id": request.doc_id if isinstance(request.doc_id, list) else [request.doc_id]
    } if request.doc_id else {}

    final_answers = []
    # Loop through each generated query object and execute queries
    for query_obj in generated_queries:
        question_text = query_obj.get("question")
        query_types = {
            "standard_query": query_obj.get("standard_query", ""),
            "fuzzy_query": query_obj.get("fuzzy_query", ""),
            "general_query": query_obj.get("general_query", "")
        }
        query_outputs = {}
        for key, query in query_types.items():
            if query:
                try:
                    query_outputs[f"{key}_output"] = run_cypher_query(query, parameters=params)
                except Exception as e:
                    query_outputs[f"{key}_output"] = {"error": str(e)}
            else:
                query_outputs[f"{key}_output"] = []

        # Build context for final LLM answer
        query_context = {
            "question": question_text,
            "standard_query": query_types["standard_query"],
            "standard_query_output": query_outputs["standard_query_output"],
            "fuzzy_query": query_types["fuzzy_query"],
            "fuzzy_query_output": query_outputs["fuzzy_query_output"],
            "general_query": query_types["general_query"],
            "general_query_output": query_outputs["general_query_output"],
            "explanation": query_obj.get("explanation", "")
        }
        answer_prompt = build_llm_answer_prompt(query_context)
        logger.debug(f"Answer prompt for question '{question_text}':\n{answer_prompt}")

        try:
            llm_answer = openai_client.call_chat_completion([
                {"role": "system",
                 "content": ("You are a professional assistant who answers queries concisely based solely on the "
                             "Neo4j cypher query outputs provided. Do not include any additional commentary.")},
                {"role": "user", "content": answer_prompt}
            ])
        except Exception as e:
            llm_answer = f"LLM error: {e}"

        final_answers.append({
            "question": question_text,
            "llm_response": llm_answer.strip()
        })

    # Build the combined response by synthesizing all individual llm_responses
    responses_for_combined = [answer["llm_response"] for answer in final_answers]
    combined_prompt = build_combined_answer_prompt(request.question, responses_for_combined)
    logger.debug(f"Combined prompt:\n{combined_prompt}")

    try:
        combined_llm_response = openai_client.call_chat_completion([
            {"role": "system",
             "content": "You are a professional assistant who synthesizes information from multiple sources."},
            {"role": "user", "content": combined_prompt}
        ])
    except Exception as e:
        combined_llm_response = f"LLM error: {e}"

    # Return a top-level response that includes the original user question,
    # the combined response, and the individual answers.
    final_response = {
        "user_question": request.question,
        "combined_response": combined_llm_response.strip(),
        "answers": final_answers
    }

    logger.debug(f"Final response: {json.dumps(final_response, indent=2)}")
    return final_response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
