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

CHUNK_SIZE_GDS = int(os.getenv("CHUNK_SIZE_GDS", "512"))

class AsyncOpenAI:
    def __init__(self, api_key: str, model: str, base_url: str, temperature: float, stop: list, timeout: int):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.stop = stop
        self.timeout = timeout

    async def invoke(self, messages: list) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stop": self.stop
        }
        logger.debug(f"LLM Payload Sent: {payload}")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
            response = await client.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return content

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
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2) + 1e-8)


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
    return "You are a professional assistant. Please provide a detailed answer."


async def extract_entities_with_llm(text: str) -> List[Dict[str, str]]:
    async_llm = get_async_llm()
    prompt = (
            "Extract all named entities from the following text. For each entity, provide its name and type (e.g., "
            "cardinal value, date value, event name, building name, geo-political entity, language name, law name, "
            "money name, person name, organization name, location name, affiliation, ordinal value, percent value, "
            "product name, quantity value, time value, name of work of art). If uncertain, label it as 'UNKNOWN'. "
            "Return ONLY a valid Python list of dictionaries (without any additional text or markdown) in the exact "
            "format as example below:"
            "[{'name': 'Entity1', 'type': 'Type1'}, {'name': 'Entity2', 'type': 'Type3'}, ...].\n\n" + text
    )
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await async_llm.invoke_json([
                {"role": "system", "content": "You are a professional NER extraction assistant who responds only in "
                                              "valid python list of dictionaries."},
                {"role": "user", "content": prompt}
            ])
            logger.debug(f"LLM response for NER extraction (attempt {attempt + 1}): {response}")

            # Try to isolate the Python list from the response.
            start = response.find('[')
            end = response.rfind(']')
            if start == -1 or end == -1:
                raise ValueError("No valid list found in response.")
            list_str = response[start:end + 1]
            try:
                entities = ast.literal_eval(list_str)
            except Exception as eval_error:
                # Fallback: try to extract using regex
                logger.error(f"ast.literal_eval failed: {eval_error}. Attempting regex extraction.")
                # This regex looks for patterns like: ['name': '...', 'type': '...']
                pattern = r"\[\s*'name':\s*'(.*?)',\s*'type':\s*'(.*?)'\s*\]"
                matches = re.findall(pattern, list_str)
                if matches:
                    entities = [{'name': name, 'type': typ} for name, typ in matches]
                else:
                    raise ValueError("Regex extraction did not find any valid entities.")

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
            await asyncio.sleep(1)  # wait a moment before retrying

async def rerank_community_reports(question: str, community_reports: list[dict]) -> list[dict]:
    """
    Uses an LLM to dynamically rerank community reports based on relevance to the user query.
    Each community report is a dict with a "summary" field.
    """
    # Create an instance of async_llm using the helper function
    async_llm = get_async_llm()

    prompt = (
        "You are an expert in evaluating textual information. Below are several community summaries. "
        "For the given user query, please rank these summaries in order of relevance from most to least relevant. "
        "For each summary, provide a JSON object containing:\n"
        "  - 'report_index': the index number (starting from 1) of the summary as listed below\n"
        "  - 'score': an integer between 1 and 100 indicating its relevance (100 being most relevant)\n"
        "Return the results as a JSON array sorted in descending order by score.\n"
        f"User Query: \"{question}\"\n"
        "Community Summaries:\n"
    )
    for idx, report in enumerate(community_reports, start=1):
        summary_text = report.get("summary", "")[:200]  # limit to first 200 characters for brevity
        prompt += f"Report {idx}: \"{summary_text}\"\n"
    prompt += (
        "\nYour JSON output should be in the format: "
        "[{\"report_index\": <number>, \"score\": <number>}, ...]"
    )

    try:
        llm_response = await async_llm.invoke([
            {"role": "system", "content": "You are an expert at ranking textual information."},
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
        return community_reports  # Fallback to original order if something goes wrong.