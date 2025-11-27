import asyncio
import re
import threading
import string
from typing import List, Dict, Any
import ast
import logging
from fastapi import HTTPException
import blingfire

logger = logging.getLogger(__name__)

import os
import json
import logging
from dotenv import load_dotenv
import httpx

logger = logging.getLogger(__name__)

# Ensure environment variables are loaded
load_dotenv()
# Load from environment
CHUNK_SIZE_GDS = int(os.getenv("CHUNK_SIZE_GDS", "512"))
COSINE_EPSILON = float(os.getenv("COSINE_EPSILON", "1e-8"))
NER_MAX_RETRIES = int(os.getenv("NER_MAX_RETRIES", "3"))
SUMMARY_TRUNCATE_CHARS = int(os.getenv("SUMMARY_TRUNCATE_CHARS", "200"))
RELEVANCE_SCORE_MAX = int(os.getenv("RELEVANCE_SCORE_MAX", "100"))
RELEVANCE_SCORE_MIN = int(os.getenv("RELEVANCE_SCORE_MIN", "1"))
# Prompt templates from .env
NER_EXTRACTION_PROMPT = os.getenv("NER_EXTRACTION_PROMPT")
NER_SYSTEM_PROMPT = os.getenv("NER_SYSTEM_PROMPT")
SYSTEM_MESSAGE_GENERIC = os.getenv("SYSTEM_MESSAGE_GENERIC")
RERANKING_PROMPT = os.getenv("RERANKING_PROMPT")
RERANKING_SYSTEM_PROMPT = os.getenv("RERANKING_SYSTEM_PROMPT")

class AsyncOpenAI:
    def __init__(self, api_key: str, model: str, base_url: str, temperature: float, stop: list, timeout: int):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.stop = stop
        self.timeout = timeout

    async def invoke(self, messages: list) -> str:
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stop": self.stop
        }
        logger.debug(f"LLM Payload Sent: {payload}")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
                    response = await client.post(self.base_url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    if not content:
                        raise ValueError("Empty response from LLM")
                    
                    return content
                    
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                
            except httpx.RequestError as e:
                logger.error(f"Request error: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))
                
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))

    async def invoke_json(self, messages: list, fallback: bool = True) -> str:
        primary_messages = [{"role": "system", "content": get_refined_system_message()}] + messages
        response_text = await self.invoke(primary_messages)
        return response_text


def get_async_llm() -> AsyncOpenAI:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "llama3.2"
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE") or "0.0")
    OPENAI_STOP = os.getenv("OPENAI_STOP")
    if OPENAI_STOP:
        OPENAI_STOP = json.loads(OPENAI_STOP)
    API_TIMEOUT = int(os.getenv("API_TIMEOUT") or "600")
    return AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        base_url=OPENAI_BASE_URL,
        temperature=OPENAI_TEMPERATURE,
        stop=OPENAI_STOP,
        timeout=API_TIMEOUT
    )


def run_async(coro):
    """
    Run an asynchronous coroutine safely even if an event loop is already running.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        result_container = {}

        def run():
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                result_container["result"] = new_loop.run_until_complete(coro)
            except Exception as e:
                result_container["result"] = e
            finally:
                new_loop.close()

        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        result = result_container.get("result")
        if isinstance(result, Exception):
            raise result
        return result
    else:
        return asyncio.run(coro)


def clean_text(text: str) -> str:
    """
    Remove non-printable characters and extra whitespace from text.
    """
    if not isinstance(text, str):
        logger.error(f"Received non-string content for text cleaning: {type(text)}")
        return ""
    cleaned_text = ''.join(filter(lambda x: x in string.printable, text.strip()))
    logger.debug(f"Cleaned text (first 100 chars): {cleaned_text[:100]}...")
    return cleaned_text


def chunk_text(text: str, max_chunk_size: int = CHUNK_SIZE_GDS) -> List[str]:
    """
    Split text into chunks up to max_chunk_size, attempting to split on sentence boundaries.
    Uses blingfire for sentence splitting.
    """
    sentences = [s.strip() for s in blingfire.text_to_sentences(text).splitlines() if s.strip()]
    unique_sentences = []
    seen = set()
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)
    chunks = []
    current_chunk = ""
    for s in unique_sentences:
        if current_chunk and (len(current_chunk) + len(s) + 1 > max_chunk_size):
            chunks.append(current_chunk.strip())
            current_chunk = s
        else:
            current_chunk = s if not current_chunk else current_chunk + " " + s
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    from numpy import dot
    from numpy.linalg import norm
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2) + COSINE_EPSILON)

def run_cypher_query(driver, query: str, parameters: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
    """
    Execute a Cypher query using the provided Neo4j driver and return a list of record data.
    """
    if parameters:
        logger.debug(f"Running Cypher query: {query} with parameters: {parameters}")
    else:
        logger.debug(f"Running Cypher query: {query}")
    try:
        with driver.session() as session:
            result = session.run(query, **parameters)
            data = [record.data() for record in result]
            logger.debug(f"Query result: {data}")
            return data
    except Exception as e:
        logger.error(f"Error executing query: {query}. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Neo4j query execution error: {e}")


def get_refined_system_message() -> str:
    """
    Return a refined system message instructing the LLM to return valid JSON.
    """
    return SYSTEM_MESSAGE_GENERIC


async def extract_entities_with_llm(text: str) -> List[Dict[str, str]]:
    if not text or not isinstance(text, str) or not text.strip():
        logger.warning("Empty, None, or invalid text provided for entity extraction")
        return []
    
    async_llm = get_async_llm()
    prompt = NER_EXTRACTION_PROMPT + text
    max_retries = NER_MAX_RETRIES
    for attempt in range(max_retries):
        try:
            response = await async_llm.invoke([
                {"role": "system", "content": NER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ])
            logger.debug(f"LLM response for NER extraction (attempt {attempt + 1}): {response}")

            start = response.find('[')
            end = response.rfind(']')
            if start == -1 or end == -1:
                raise ValueError("No valid list found in response.")
            list_str = response[start:end + 1]
            # Enhanced parsing with multiple fallback strategies
            entities = parse_entities_from_response_enhanced(list_str)

            if not isinstance(entities, list):
                raise ValueError("Extracted value is not a list.")
            filtered_entities = [
                entity for entity in entities
                if isinstance(entity, dict) and entity.get('name', '').strip() and entity.get('type', '').strip()
            ]
            return filtered_entities
        except Exception as e:
            logger.error(f"NER extraction attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return []
            await asyncio.sleep(1)

def parse_entities_from_response_enhanced(response: str) -> List[Dict[str, str]]:
    """Parse entities from LLM response with multiple fallback strategies"""
    
    # Strategy 1: Direct JSON parsing
    try:
        import json
        # Look for JSON array in response
        start = response.find('[')
        end = response.rfind(']')
        if start != -1 and end != -1:
            json_str = response[start:end + 1]
            entities = json.loads(json_str)
            if isinstance(entities, list):
                return validate_and_filter_entities_enhanced(entities)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: AST literal eval
    try:
        entities = ast.literal_eval(response)
        if isinstance(entities, list):
            return validate_and_filter_entities_enhanced(entities)
    except (ValueError, SyntaxError):
        pass
    
    # Strategy 3: Enhanced regex extraction for structured format
    try:
        # Pattern for {"name": "...", "type": "..."}
        pattern = r'\{\s*["\']name["\']\s*:\s*["\']([^"\']+)["\']\s*,\s*["\']type["\']\s*:\s*["\']([^"\']+)["\']\s*\}'
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            entities = [{'name': name.strip(), 'type': type_.strip()} for name, type_ in matches]
            return validate_and_filter_entities_enhanced(entities)
    except Exception:
        pass
    
    # Strategy 4: Original regex pattern as fallback
    try:
        pattern = r"\[\s*'name':\s*'(.*?)',\s*'type':\s*'(.*?)'\s*\]"
        matches = re.findall(pattern, response)
        if matches:
            entities = [{'name': name, 'type': typ} for name, typ in matches]
            return validate_and_filter_entities_enhanced(entities)
    except Exception:
        pass
    
    logger.warning("All entity parsing strategies failed, returning empty list")
    return []

def validate_and_filter_entities_enhanced(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Validate and filter entities with enhanced criteria"""
    if not isinstance(entities, list):
        return []
    
    valid_types = {
        'ORGANIZATION', 'TECHNOLOGY', 'PRODUCT', 'PERSON', 'LOCATION', 
        'CONCEPT', 'EVENT', 'DATE', 'METRIC', 'UNKNOWN',
        # Also accept some common variations for backward compatibility
        'Organization', 'Framework', 'Database', 'Model', 'Entity', 'Relationship'
    }
    
    filtered = []
    seen_names = set()
    
    for entity in entities:
        if not isinstance(entity, dict):
            continue
            
        name = entity.get('name', '').strip()
        entity_type = entity.get('type', '').strip()
        
        # Enhanced validation criteria
        if (name and entity_type and 
            len(name) >= 2 and 
            len(name) <= 100 and
            entity_type in valid_types and
            name.lower() not in seen_names and
            not name.isdigit() and
            not re.match(r'^[^\w\s]+$', name)):  # Not just punctuation
            
            # Normalize the name
            normalized_name = ' '.join(name.split())  # Clean whitespace
            
            # Normalize entity type to standard format
            if entity_type in ['Organization', 'Entity']:
                entity_type = 'ORGANIZATION'
            elif entity_type in ['Framework', 'Database', 'Model']:
                entity_type = 'TECHNOLOGY'
            elif entity_type == 'Relationship':
                entity_type = 'CONCEPT'
            
            filtered.append({
                'name': normalized_name,
                'type': entity_type.upper()
            })
            seen_names.add(normalized_name.lower())
    
    return filtered

async def rerank_community_reports(question: str, community_reports: list[dict]) -> list[dict]:
    async_llm = get_async_llm()

    prompt = RERANKING_PROMPT.format(
        question=question,
        RELEVANCE_SCORE_MIN=RELEVANCE_SCORE_MIN,
        RELEVANCE_SCORE_MAX=RELEVANCE_SCORE_MAX
    )
    for idx, report in enumerate(community_reports, start=1):
        summary_text = report.get("summary", "")[:SUMMARY_TRUNCATE_CHARS]
        prompt += f"Report {idx}: \"{summary_text}\"\n"
    prompt += (
        "\nYour JSON output should be in the format: "
        "[{\"report_index\": <number>, \"score\": <number>}, ...]"
    )

    try:
        llm_response = await async_llm.invoke([
            {"role": "system", "content": RERANKING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])
        ranking = json.loads(llm_response)
        ranking = [{"index": item["report_index"] - 1, "score": item["score"]} for item in ranking if
                   "report_index" in item and "score" in item]
        ranking.sort(key=lambda x: x["score"], reverse=True)
        sorted_reports = [community_reports[item["index"]] for item in ranking if
                          item["index"] < len(community_reports)]
        return sorted_reports
    except Exception as e:
        logger.error(f"Error during LLM-based reranking of community reports: {e}")
        return community_reports
