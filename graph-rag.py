import os
import re
import json
import time
import string
import uuid
import logging
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

import requests
import blingfire
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

from neo4j import GraphDatabase

# Import Text2CypherRetriever from neo4j_graphrag package
from neo4j_graphrag.retrievers.text2cypher import Text2CypherRetriever

# For NER extraction using Flair
from langdetect import detect
from flair.models import SequenceTagger
from flair.data import Sentence

import PyPDF2
import docx

# -----------------------------------------------------------------------------
# Load Environment Variables and Setup Logging
# -----------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("GraphRAG")

# Global constant for the projected graph name.
GRAPH_NAME = os.getenv("GRAPH_NAME", "entityGraph")

# -----------------------------------------------------------------------------
# Neo4j Configuration & Driver Initialization
# -----------------------------------------------------------------------------
NEO4J_URI = os.getenv("DB_URL", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("DB_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("DB_PASSWORD", "neo4j")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# -----------------------------------------------------------------------------
# OpenAI LLM Class with Base URL (implements invoke)
# -----------------------------------------------------------------------------
class OpenAI:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo",
                 base_url: str = "https://api.openai.com/v1/chat/completions", temperature: float = 0.0):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(self.base_url, json=payload, headers=headers, timeout=60, verify=False)
        response.raise_for_status()
        output = response.json()["choices"][0]["message"]["content"]
        return output.strip()

# -----------------------------------------------------------------------------
# Helper Functions for Neo4j Queries and Graph Statistics
# -----------------------------------------------------------------------------
def clean_json_response(response: str) -> str:
    response = response.strip()
    # Remove markdown code fences if present
    if response.startswith("```") and response.endswith("```"):
        lines = response.splitlines()
        if len(lines) >= 3:
            response = "\n".join(lines[1:-1])
        else:
            response = ""
    return response
    
def run_cypher_query(query: str, parameters: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
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
    CALL apoc.meta.schema({ includeLabels: true, includeRels: true, sample: -1 })
    YIELD value
    UNWIND keys(value) as label
    WITH label, value[label] as data
    WHERE data.type = "node"
    MATCH (n)
    OPTIONAL MATCH (n)-[r]->(m)
    WITH label, data, collect(DISTINCT { type: type(r), direction: "outgoing", targetLabel: head(labels(m)) }) as outRelations
    OPTIONAL MATCH (n)<-[r]-(m)
    WITH label, data, outRelations, collect(DISTINCT { type: type(r), direction: "incoming", sourceLabel: head(labels(m)) }) as inRelations
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

# -----------------------------------------------------------------------------
# LLM Prompt Builders (for answer synthesis and combined response)
# -----------------------------------------------------------------------------
def build_llm_answer_prompt(query_context: dict) -> str:
    prompt = f"""Given the data extracted from the graph database:

Question: {query_context.get('question')}

Standard Query Output:
{json.dumps(query_context.get('standard_query_output'), indent=2)}

Fuzzy Query Output:
{json.dumps(query_context.get('fuzzy_query_output'), indent=2)}

General Query Output:
{json.dumps(query_context.get('general_query_output'), indent=2)}

Provide a detailed answer to the query based solely on the information above."""
    return prompt.strip()

def build_combined_answer_prompt(user_question: str, responses: List[str]) -> str:
    prompt = f"User Question: {user_question}\n\nThe following responses were generated for different aspects of your query:\n"
    for idx, resp in enumerate(responses, start=1):
        prompt += f"{idx}. {resp}\n"
    prompt += "\nBased on the above responses, provide a final comprehensive answer that directly addresses the original question."
    return prompt.strip()

# -----------------------------------------------------------------------------
# Global Search Map-Reduce Implementation
# -----------------------------------------------------------------------------
def global_search_map_reduce(question: str, conversation_history: Optional[str] = None, chunk_size: int = 512, top_n: int = 5) -> str:
    # Generate community summaries (for all communities in the dataset)
    summaries = generate_community_summaries(doc_id=None)
    # Extract community report texts and shuffle them
    community_reports = list(summaries.values())
    random.shuffle(community_reports)
    
    # Map step: Process each community report in chunks to extract intermediate key points with ratings.
    llm_instance = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                          model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                          base_url=os.getenv("OPENAI_BASE_URL"),
                          temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")))
    intermediate_points = []
    for report in community_reports:
        chunks = chunk_text(report, max_chunk_size=chunk_size)
        for chunk in chunks:
            map_prompt = f"""
You are an expert in extracting key information. Given the following community report chunk and the user query, extract key points that are relevant for answering the query. For each key point, provide a numerical rating (between 1 and 100) indicating its importance.
Output your answer in JSON format as a list of objects, each with keys "point" and "rating".

Community Report Chunk:
\"\"\"{chunk}\"\"\"

User Query:
\"\"\"{question}\"\"\"
            """.strip()
            try:
                map_response = llm_instance.invoke([
                    {"role": "system", "content": "You are a professional extraction assistant."},
                    {"role": "user", "content": map_prompt}
                ])
                points = json.loads(map_response)
                if isinstance(points, list):
                    intermediate_points.extend(points)
            except Exception as e:
                logger.error(f"Map step error for chunk: {e}")
    
    # Reduce step: Aggregate and filter the intermediate points.
    intermediate_points_sorted = sorted(intermediate_points, key=lambda x: x.get("rating", 0), reverse=True)
    selected_points = intermediate_points_sorted[:top_n]
    aggregated_context = "\n".join([f"{pt['point']} (Rating: {pt['rating']})" for pt in selected_points])
    
    conv_text = f"Conversation History: {conversation_history}\n" if conversation_history else ""
    reduce_prompt = f"""
You are a professional synthesizer. Given the following aggregated key points extracted from community reports and the user query, generate a comprehensive final answer.
Aggregated Key Points:
\"\"\"{aggregated_context}\"\"\"

{conv_text}User Query:
\"\"\"{question}\"\"\"
    """.strip()
    try:
        final_answer = llm_instance.invoke([
            {"role": "system", "content": "You are a professional assistant skilled in synthesizing information."},
            {"role": "user", "content": reduce_prompt}
        ])
    except Exception as e:
        final_answer = f"Error during reduce step: {e}"
    return final_answer

# -----------------------------------------------------------------------------
# Helper Functions for Local and Community Context (for Local/DRIFT search)
# -----------------------------------------------------------------------------
def get_local_context(entity: str) -> str:
    query = """
    MATCH (e:Entity {name: $entity})
    OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
    RETURN e.name AS entity, collect({neighbor: neighbor.name, relationship: type(r)}) as neighbors
    """
    result = run_cypher_query(query, {"entity": entity.lower()})
    if result:
        row = result[0]
        context = f"Entity: {row['entity']}\n"
        if row.get('neighbors'):
            neighbors_list = [f"{item['neighbor']} ({item.get('relationship', '')})" for item in row['neighbors'] if item.get('neighbor')]
            context += "Neighbors: " + ", ".join(neighbors_list) + "\n"
        else:
            context += "No neighbors found.\n"
        return context
    return "No local context found."

def get_community_context(entity: str) -> str:
    communities = detect_communities(doc_id=None)
    community_for_entity = None
    for comm, entities in communities.items():
        if entity.lower() in [e.lower() for e in entities]:
            community_for_entity = comm
            break
    if community_for_entity is not None:
        summaries = generate_community_summaries(doc_id=None)
        summary = summaries.get(community_for_entity, "No summary available.")
        return f"Community {community_for_entity} summary: {summary}"
    return "Entity not found in any community."

# -----------------------------------------------------------------------------
# Pydantic Models for Endpoints
# -----------------------------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str
    doc_id: Optional[Union[str, List[str]]] = None
    previous_conversations: Optional[str] = None

class DeleteDocumentRequest(BaseModel):
    doc_id: Optional[str] = None
    document_name: Optional[str] = None

class LocalSearchRequest(BaseModel):
    entity: str
    question: str

class DriftSearchRequest(BaseModel):
    entity: str
    question: str

# -----------------------------------------------------------------------------
# NER Extraction using Flair
# -----------------------------------------------------------------------------
model_cache = {}
DEFAULT_MODEL_MAP = {
    'en': 'ner-large',
    'de': 'de-ner-large',
    'es': 'es-ner-large',
    'nl': 'nl-ner-large',
    'fr': 'fr-ner',
    'da': 'da-ner',
    'ar': 'ar-ner',
    'uk': 'ner-ukrainian'
}

def extract_flair_entities(text: str) -> List[Dict[str, str]]:
    try:
        lang = detect(text).lower()
    except Exception:
        lang = 'en'
    if '-' in lang:
        lang = lang.split('-')[0]
    model_name = DEFAULT_MODEL_MAP.get(lang, 'ner-large')
    if model_name in model_cache:
        tagger = model_cache[model_name]
    else:
        try:
            tagger = SequenceTagger.load(model_name)
            model_cache[model_name] = tagger
        except Exception as e:
            raise Exception(f"Failed to load model {model_name}: {e}")
    sentence_strs = [s.strip() for s in blingfire.text_to_sentences(text).splitlines() if s.strip()]
    sentences = [Sentence(s) for s in sentence_strs]
    try:
        tagger.predict(sentences, mini_batch_size=32)
    except Exception as e:
        raise Exception(f"NER prediction error: {e}")
    entities = {}
    for sentence in sentences:
        for label in sentence.get_labels():
            key = (label.data_point.text, label.value)
            entities[key] = {"name": label.data_point.text, "label": label.value}
    return list(entities.values())

# -----------------------------------------------------------------------------
# Document Processing: Clean Text and Chunking
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    return ''.join(filter(lambda x: x in string.printable, text.strip()))

def chunk_text(text: str, max_chunk_size: int = 512) -> List[str]:
    max_chunk_size = int(max_chunk_size)
    sentences = [s.strip() for s in blingfire.text_to_sentences(text).splitlines() if s.strip()]
    chunks = []
    current_chunk = ""
    for s in sentences:
        if current_chunk and (len(current_chunk) + len(s) + 1) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = s
        else:
            current_chunk = s if not current_chunk else current_chunk + " " + s
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# -----------------------------------------------------------------------------
# Graph Manager: Build Graph (Centrality functions removed)
# -----------------------------------------------------------------------------
class GraphManager:
    def __init__(self):
        pass

    def _merge_entity(self, session, name: str, doc_id: str, chunk_id: int):
        query = "MERGE (e:Entity {name: $name, doc_id: $doc_id}) MERGE (e)-[:MENTIONED_IN]->(c:Chunk {id: $cid, doc_id: $doc_id})"
        session.run(query, name=name, doc_id=doc_id, cid=chunk_id)

    def _merge_cooccurrence(self, session, name_a: str, name_b: str, doc_id: str):
        query = ("MATCH (a:Entity {name: $name_a, doc_id: $doc_id}), "
                 "(b:Entity {name: $name_b, doc_id: $doc_id}) "
                 "MERGE (a)-[:CO_OCCURS_WITH]->(b)")
        session.run(query, name_a=name_a, name_b=name_b, doc_id=doc_id)

    def build_graph(self, chunks: List[str], metadata_list: List[Dict[str, Any]]) -> None:
        with driver.session() as session:
            for i, (chunk, meta) in enumerate(zip(chunks, metadata_list)):
                query = """
                    MERGE (c:Chunk {id: $cid, doc_id: $doc_id})
                    ON CREATE SET c.text = $text, c.document_name = $document_name, c.timestamp = $timestamp
                    ON MATCH SET c.text = $text, c.document_name = $document_name, c.timestamp = $timestamp
                """
                session.run(query, cid=i, doc_id=meta["doc_id"], text=chunk,
                            document_name=meta.get("document_name"), timestamp=meta.get("timestamp"))
                try:
                    entities = extract_flair_entities(chunk)
                except Exception as e:
                    logger.error(f"NER extraction error on chunk {i}: {e}")
                    entities = []
                names = [e["name"].strip().lower() for e in entities]
                for name in names:
                    self._merge_entity(session, name, meta["doc_id"], i)
                if len(names) > 1:
                    for j in range(len(names)):
                        for k in range(j + 1, len(names)):
                            self._merge_cooccurrence(session, names[j], names[k], meta["doc_id"])
                try:
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                                    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                                    base_url=os.getenv("OPENAI_BASE_URL"),
                                    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")))
                    summary = client.invoke([
                        {"role": "system", "content": "Summarize the following text into relationships in the format: Entity1 -> Entity2 [strength: X.X]. Do not add extra commentary."},
                        {"role": "user", "content": chunk}
                    ])
                except Exception as e:
                    logger.error(f"Summarization error on chunk {i}: {e}")
                    summary = ""
                for line in summary.split("\n"):
                    if "->" in line:
                        parts = line.split("->")
                        if len(parts) >= 2:
                            source = parts[0].strip().lower()
                            target = parts[-1].split("[")[0].strip().lower()
                            weight = 1.0
                            match = re.search(r"\[strength:\s*(\d\.\d)\]", line)
                            if match:
                                weight = float(match.group(1))
                            rel_query = """
                                MERGE (a:Entity {name: $source, doc_id: $doc_id})
                                MERGE (b:Entity {name: $target, doc_id: $doc_id})
                                MERGE (a)-[r:RELATES_TO {weight: $weight}]->(b)
                            """
                            session.run(rel_query, source=source, target=target, weight=weight, doc_id=meta["doc_id"])
        logger.info("Graph construction complete.")

    def reproject_graph(self, graph_name: str = GRAPH_NAME, doc_id: Optional[str] = None) -> None:
        with driver.session() as session:
            exists_record = session.run("CALL gds.graph.exists($graph_name) YIELD exists", graph_name=graph_name).single()
            if exists_record and exists_record["exists"]:
                session.run("CALL gds.graph.drop($graph_name)", graph_name=graph_name)
            if doc_id:
                node_query = f"MATCH (n:Entity) WHERE n.doc_id = '{doc_id}' RETURN id(n) AS id"
                rel_query = f"MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE a.doc_id = '{doc_id}' AND b.doc_id = '{doc_id}' RETURN id(a) AS source, id(b) AS target, r.weight AS weight"
                config = { "relationshipProjection": { "RELATES_TO": { "orientation": "UNDIRECTED" } } }
                session.run(
                    "CALL gds.graph.project.cypher($graph_name, $nodeQuery, $relQuery, $config)",
                    graph_name=graph_name, nodeQuery=node_query, relQuery=rel_query, config=config
                )
            else:
                config = { "RELATES_TO": { "orientation": "UNDIRECTED" } }
                session.run(
                    "CALL gds.graph.project($graph_name, ['Entity'], $config)",
                    graph_name=graph_name, config=config
                )
            logger.info("Graph projection complete.")

# Instantiate GraphManager
graph_manager = GraphManager()

# -----------------------------------------------------------------------------
# Community Detection and Summarization Functions
# -----------------------------------------------------------------------------
def detect_communities(doc_id: Optional[str] = None) -> dict:
    communities = {}
    with driver.session() as session:
        exists_record = session.run("CALL gds.graph.exists($graph_name) YIELD exists", graph_name=GRAPH_NAME).single()
        if exists_record and exists_record["exists"]:
            session.run("CALL gds.graph.drop($graph_name)", graph_name=GRAPH_NAME)
        if doc_id:
            node_query = f"MATCH (n:Entity) WHERE n.doc_id = '{doc_id}' RETURN id(n) AS id"
            rel_query = f"MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE a.doc_id = '{doc_id}' AND b.doc_id = '{doc_id}' RETURN id(a) AS source, id(b) AS target, r.weight AS weight"
            config = { "relationshipProjection": { "RELATES_TO": { "orientation": "UNDIRECTED" } } }
            session.run(
                "CALL gds.graph.project.cypher($graph_name, $nodeQuery, $relQuery, $config)",
                graph_name=GRAPH_NAME, nodeQuery=node_query, relQuery=rel_query, config=config
            )
        else:
            config = { "RELATES_TO": { "orientation": "UNDIRECTED" } }
            session.run(
                "CALL gds.graph.project($graph_name, ['Entity'], $config)",
                graph_name=GRAPH_NAME, config=config
            )
        result = session.run(
            "CALL gds.leiden.stream($graph_name) YIELD nodeId, communityId "
            "RETURN gds.util.asNode(nodeId).name AS entity, communityId AS community ORDER BY community, entity",
            graph_name=GRAPH_NAME
        )
        for record in result:
            comm = record["community"]
            entity = record["entity"]
            communities.setdefault(comm, []).append(entity)
        session.run("CALL gds.graph.drop($graph_name)", graph_name=GRAPH_NAME)
    return communities

def generate_community_summaries(doc_id: Optional[str] = None) -> dict:
    communities = detect_communities(doc_id)
    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                 model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                 base_url=os.getenv("OPENAI_BASE_URL"),
                 temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")))
    summaries = {}
    for comm, entities in communities.items():
        prompt = f"Summarize the following community of entities: {', '.join(entities)}. Provide a concise summary capturing the main themes and key relationships."
        summary = llm.invoke([
            {"role": "system", "content": "You are a professional summarization assistant."},
            {"role": "user", "content": prompt}
        ])
        summaries[comm] = summary
    return summaries

# -----------------------------------------------------------------------------
# Global Search Map-Reduce Implementation
# -----------------------------------------------------------------------------
def global_search_map_reduce(question: str, conversation_history: Optional[str] = None, chunk_size: int = 512, top_n: int = 5) -> str:
    # Generate community summaries
    summaries = generate_community_summaries(doc_id=None)
    community_reports = list(summaries.values())
    random.shuffle(community_reports)
    
    llm_instance = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                          model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                          base_url=os.getenv("OPENAI_BASE_URL"),
                          temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")))
    intermediate_points = []
    for report in community_reports:
        chunks = chunk_text(report, max_chunk_size=chunk_size)
        for chunk in chunks:
            map_prompt = f"""
You are an expert in extracting key information. Given the following community report chunk and the user query, extract key points that are relevant for answering the query.
For each key point, output an object with two keys: "point" (a short text string) and "rating" (a numerical value between 1 and 100 indicating its importance).
Return only a valid JSON array (and nothing else).

Community Report Chunk:
\"\"\"{chunk}\"\"\"

User Query:
\"\"\"{question}\"\"\"
            """.strip()
            try:
                map_response = llm_instance.invoke([
                    {"role": "system", "content": "You are a professional extraction assistant."},
                    {"role": "user", "content": map_prompt}
                ])
                if not map_response.strip():
                    logger.warning("LLM returned an empty response for this chunk. Skipping.")
                    continue
                # Clean the response to remove markdown formatting
                cleaned_response = clean_json_response(map_response)
                try:
                    points = json.loads(cleaned_response)
                except Exception as json_e:
                    logger.error(f"JSON parsing error: {json_e}. Raw response after cleaning: {cleaned_response}")
                    continue  # Skip this chunk if JSON parsing fails
                if isinstance(points, list):
                    intermediate_points.extend(points)
            except Exception as e:
                logger.error(f"Map step error for chunk: {e}")
    
    intermediate_points_sorted = sorted(intermediate_points, key=lambda x: x.get("rating", 0), reverse=True)
    selected_points = intermediate_points_sorted[:top_n]
    aggregated_context = "\n".join([f"{pt['point']} (Rating: {pt['rating']})" for pt in selected_points])
    
    conv_text = f"Conversation History: {conversation_history}\n" if conversation_history else ""
    reduce_prompt = f"""
You are a professional synthesizer. Given the following aggregated key points extracted from community reports and the user query, generate a comprehensive final answer.
Aggregated Key Points:
\"\"\"{aggregated_context}\"\"\"

{conv_text}User Query:
\"\"\"{question}\"\"\"
    """.strip()
    try:
        final_answer = llm_instance.invoke([
            {"role": "system", "content": "You are a professional assistant skilled in synthesizing information."},
            {"role": "user", "content": reduce_prompt}
        ])
    except Exception as e:
        final_answer = f"Error during reduce step: {e}"
    return final_answer

# -----------------------------------------------------------------------------
# Helper Functions for Local and Community Context (for Local/DRIFT search)
# -----------------------------------------------------------------------------
def get_local_context(entity: str) -> str:
    query = """
    MATCH (e:Entity {name: $entity})
    OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
    RETURN e.name AS entity, collect({neighbor: neighbor.name, relationship: type(r)}) as neighbors
    """
    result = run_cypher_query(query, {"entity": entity.lower()})
    if result:
        row = result[0]
        context = f"Entity: {row['entity']}\n"
        if row.get('neighbors'):
            neighbors_list = [f"{item['neighbor']} ({item.get('relationship', '')})" for item in row['neighbors'] if item.get('neighbor')]
            context += "Neighbors: " + ", ".join(neighbors_list) + "\n"
        else:
            context += "No neighbors found.\n"
        return context
    return "No local context found."

def get_community_context(entity: str) -> str:
    communities = detect_communities(doc_id=None)
    community_for_entity = None
    for comm, entities in communities.items():
        if entity.lower() in [e.lower() for e in entities]:
            community_for_entity = comm
            break
    if community_for_entity is not None:
        summaries = generate_community_summaries(doc_id=None)
        summary = summaries.get(community_for_entity, "No summary available.")
        return f"Community {community_for_entity} summary: {summary}"
    return "Entity not found in any community."

# -----------------------------------------------------------------------------
# Pydantic Models for Endpoints (Request Bodies)
# -----------------------------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str
    doc_id: Optional[Union[str, List[str]]] = None
    previous_conversations: Optional[str] = None

class DeleteDocumentRequest(BaseModel):
    doc_id: Optional[str] = None
    document_name: Optional[str] = None

class LocalSearchRequest(BaseModel):
    entity: str
    question: str

class DriftSearchRequest(BaseModel):
    entity: str
    question: str

# -----------------------------------------------------------------------------
# NER Extraction using Flair
# -----------------------------------------------------------------------------
model_cache = {}
DEFAULT_MODEL_MAP = {
    'en': 'ner-large',
    'de': 'de-ner-large',
    'es': 'es-ner-large',
    'nl': 'nl-ner-large',
    'fr': 'fr-ner',
    'da': 'da-ner',
    'ar': 'ar-ner',
    'uk': 'ner-ukrainian'
}

def extract_flair_entities(text: str) -> List[Dict[str, str]]:
    try:
        lang = detect(text).lower()
    except Exception:
        lang = 'en'
    if '-' in lang:
        lang = lang.split('-')[0]
    model_name = DEFAULT_MODEL_MAP.get(lang, 'ner-large')
    if model_name in model_cache:
        tagger = model_cache[model_name]
    else:
        try:
            tagger = SequenceTagger.load(model_name)
            model_cache[model_name] = tagger
        except Exception as e:
            raise Exception(f"Failed to load model {model_name}: {e}")
    sentence_strs = [s.strip() for s in blingfire.text_to_sentences(text).splitlines() if s.strip()]
    sentences = [Sentence(s) for s in sentence_strs]
    try:
        tagger.predict(sentences, mini_batch_size=32)
    except Exception as e:
        raise Exception(f"NER prediction error: {e}")
    entities = {}
    for sentence in sentences:
        for label in sentence.get_labels():
            key = (label.data_point.text, label.value)
            entities[key] = {"name": label.data_point.text, "label": label.value}
    return list(entities.values())

# -----------------------------------------------------------------------------
# Document Processing: Clean Text and Chunking
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    return ''.join(filter(lambda x: x in string.printable, text.strip()))

def chunk_text(text: str, max_chunk_size: int = 512) -> List[str]:
    max_chunk_size = int(max_chunk_size)
    sentences = [s.strip() for s in blingfire.text_to_sentences(text).splitlines() if s.strip()]
    chunks = []
    current_chunk = ""
    for s in sentences:
        if current_chunk and (len(current_chunk) + len(s) + 1) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = s
        else:
            current_chunk = s if not current_chunk else current_chunk + " " + s
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# -----------------------------------------------------------------------------
# FastAPI Application and Endpoints
# -----------------------------------------------------------------------------
app = FastAPI(title="Graph RAG API", description="End-to-end Graph Database RAG on Neo4j", version="1.0.0")

# Endpoint: Upload Documents and Build Graph
@app.post("/upload_documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    document_texts = []
    metadata = []
    for file in files:
        filename = file.filename.lower()
        try:
            if filename.endswith(".pdf"):
                reader = PyPDF2.PdfReader(file.file)
                text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
            elif filename.endswith(".docx"):
                doc = docx.Document(file.file)
                text = "\n".join(para.text for para in doc.paragraphs)
            elif filename.endswith(".txt"):
                text = (await file.read()).decode("utf-8")
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.filename}")
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}")
        text = clean_text(text)
        doc_id = str(uuid.uuid4())
        metadata.append({
            "doc_id": doc_id,
            "document_name": file.filename,
            "timestamp": datetime.now().isoformat()
        })
        document_texts.append(text)
    all_chunks = []
    meta_list = []
    for text, meta in zip(document_texts, metadata):
        chunks = chunk_text(text, max_chunk_size=int(os.getenv("CHUNK_SIZE_GDS", "1024")))
        all_chunks.extend(chunks)
        meta_list.extend([meta] * len(chunks))
    try:
        graph_manager.build_graph(all_chunks, meta_list)
    except Exception as e:
        logger.error(f"Graph building error: {e}")
        raise HTTPException(status_code=500, detail=f"Graph building error: {e}")
    return {"message": "Documents processed and graph updated successfully."}

# Endpoint: Ask Question (with previous conversation rewriting and using Text2Cypher)
@app.post("/ask_question")
def ask_question(request: QuestionRequest):
    llm_instance = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                          model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                          base_url=os.getenv("OPENAI_BASE_URL"),
                          temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")))
    if request.previous_conversations:
        rewrite_prompt = (
            f"Based on the user's previous history query about '{request.previous_conversations}', "
            f"rewrite their new query: '{request.question}' into a standalone query."
        )
        rewritten_query = llm_instance.invoke([
            {"role": "system", "content": "You are a professional query rewriting assistant."},
            {"role": "user", "content": rewrite_prompt}
        ])
        logger.debug(f"Rewritten query: {rewritten_query}")
        request.question = rewritten_query
    retriever = Text2CypherRetriever(driver, llm_instance)
    try:
        generated_queries = retriever.get_cypher(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating cypher queries: {e}")
    final_answers = []
    for item in generated_queries:
        question_text = item.get("question", "")
        query_types = {
            "standard_query": item.get("standard_query", ""),
            "fuzzy_query": item.get("fuzzy_query", ""),
            "general_query": item.get("general_query", "")
        }
        query_outputs = {}
        for key, query in query_types.items():
            if query:
                try:
                    query_outputs[f"{key}_output"] = run_cypher_query(query)
                except Exception as e:
                    query_outputs[f"{key}_output"] = {"error": str(e)}
            else:
                query_outputs[f"{key}_output"] = []
        query_context = {
            "question": question_text,
            "standard_query": query_types["standard_query"],
            "standard_query_output": query_outputs["standard_query_output"],
            "fuzzy_query": query_types["fuzzy_query"],
            "fuzzy_query_output": query_outputs["fuzzy_query_output"],
            "general_query": query_types["general_query"],
            "general_query_output": query_outputs["general_query_output"],
            "explanation": item.get("explanation", "")
        }
        answer_prompt = build_llm_answer_prompt(query_context)
        logger.debug(f"Answer prompt for question '{question_text}':\n{answer_prompt}")
        try:
            llm_answer = llm_instance.invoke([
                {"role": "system", "content": "You are a professional assistant who answers queries concisely based solely on the provided Neo4j Cypher query outputs."},
                {"role": "user", "content": answer_prompt}
            ])
        except Exception as e:
            llm_answer = f"LLM error: {e}"
        final_answers.append({
            "question": question_text,
            "llm_response": llm_answer.strip()
        })
    responses_for_combined = [answer["llm_response"] for answer in final_answers]
    combined_prompt = build_combined_answer_prompt(request.question, responses_for_combined)
    logger.debug(f"Combined prompt:\n{combined_prompt}")
    try:
        combined_llm_response = llm_instance.invoke([
            {"role": "system", "content": "You are a professional assistant who synthesizes information from multiple sources to provide a detailed final answer."},
            {"role": "user", "content": combined_prompt}
        ])
    except Exception as e:
        combined_llm_response = f"LLM error: {e}"
    final_response = {
        "user_question": request.question,
        "combined_response": combined_llm_response.strip(),
        "answers": final_answers
    }
    logger.debug(f"Final response: {json.dumps(final_response, indent=2)}")
    return final_response

# Endpoint: Global Search – using map-reduce on community summaries
@app.post("/global_search")
def global_search(request: QuestionRequest):
    answer = global_search_map_reduce(request.question, conversation_history=request.previous_conversations, chunk_size=512, top_n=5)
    return {"global_search_answer": answer}

# Endpoint: Local Search – reasoning about a specific entity via its neighbors.
@app.post("/local_search")
def local_search(request: LocalSearchRequest):
    llm_instance = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                          model=os.getenv("OPENAI_MODEL"),
                          base_url=os.getenv("OPENAI_BASE_URL"),
                          temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")))
    context = get_local_context(request.entity)
    prompt = f"Based on the following local context:\n{context}\n\nAnswer the following question about the entity '{request.entity}': {request.question}"
    try:
        answer = llm_instance.invoke([
            {"role": "system", "content": "You are a professional assistant skilled in reasoning based on local context information."},
            {"role": "user", "content": prompt}
        ])
    except Exception as e:
        answer = f"LLM error: {e}"
    return {"local_search_answer": answer}

# Endpoint: DRIFT Search – local search with additional community context.
@app.post("/drift_search")
def drift_search(request: DriftSearchRequest):
    llm_instance = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                          model=os.getenv("OPENAI_MODEL"),
                          base_url=os.getenv("OPENAI_BASE_URL"),
                          temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")))
    local_context_text = get_local_context(request.entity)
    community_context_text = get_community_context(request.entity)
    prompt = f"Based on the following information:\nLocal Context:\n{local_context_text}\n\nCommunity Context:\n{community_context_text}\n\nAnswer the following question about the entity '{request.entity}': {request.question}"
    try:
        answer = llm_instance.invoke([
            {"role": "system", "content": "You are a professional assistant skilled in synthesizing local and community context to provide detailed answers."},
            {"role": "user", "content": prompt}
        ])
    except Exception as e:
        answer = f"LLM error: {e}"
    return {"drift_search_answer": answer}

# Endpoint: List Documents with Metadata
@app.get("/documents")
def list_documents():
    try:
        query = """
            MATCH (c:Chunk)
            RETURN DISTINCT c.doc_id AS doc_id, c.document_name AS document_name, c.timestamp AS timestamp
        """
        results = run_cypher_query(query)
        return {"documents": results}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Error listing documents")

# Endpoint: Delete a Document based on doc_id or document_name
@app.delete("/delete_document")
def delete_document(request: DeleteDocumentRequest):
    if not request.doc_id and not request.document_name:
        raise HTTPException(status_code=400, detail="Provide either doc_id or document_name to delete a document.")
    try:
        if request.doc_id:
            query = "MATCH (n) WHERE n.doc_id = $doc_id DETACH DELETE n"
            run_cypher_query(query, {"doc_id": request.doc_id})
            message = f"Document with doc_id {request.doc_id} deleted successfully."
        else:
            query = "MATCH (n) WHERE n.document_name = $document_name DETACH DELETE n"
            run_cypher_query(query, {"document_name": request.document_name})
            message = f"Document with name {request.document_name} deleted successfully."
        return {"message": message}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {e}")

# Endpoint: Get Communities (using Leiden algorithm)
@app.get("/communities")
def get_communities(doc_id: Optional[str] = None):
    try:
        communities = detect_communities(doc_id)
        return {"communities": communities}
    except Exception as e:
        logger.error(f"Error detecting communities: {e}")
        raise HTTPException(status_code=500, detail=f"Error detecting communities: {e}")

# Endpoint: Get Community Summaries
@app.get("/community_summaries")
def community_summaries(doc_id: Optional[str] = None):
    try:
        summaries = generate_community_summaries(doc_id)
        return {"community_summaries": summaries}
    except Exception as e:
        logger.error(f"Error generating community summaries: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating community summaries: {e}")

# -----------------------------------------------------------------------------
# Main: Run the Application
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
