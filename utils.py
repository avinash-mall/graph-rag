"""
Utilities for Graph RAG with LLM-based NLP preprocessing and optimized performance.

Features:
- LLM-based NER using gemma3:1b model via OpenAI-compatible API
- LLM-based coreference resolution using gemma3:1b model
- Boilerplate and navigation text removal before chunking
- Batch embedding processing with intelligent caching
- Async optimization throughout
- Comprehensive error handling and logging
"""

import asyncio
import hashlib
import logging
import os
import re
import string
import threading
from typing import List, Dict, Any, Optional, Tuple, Union
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

import httpx
import numpy as np
import blingfire
from dotenv import load_dotenv
from fastapi import HTTPException

# Load environment variables
load_dotenv()

# Configuration
CHUNK_SIZE_GDS = int(os.getenv("CHUNK_SIZE_GDS", "512"))
COSINE_EPSILON = float(os.getenv("COSINE_EPSILON", "1e-8"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Setup logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("Utils")

# Global thread pool for CPU-bound tasks
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

@dataclass
class Entity:
    """Structured entity representation"""
    name: str
    type: str
    confidence: float = 0.0
    start_pos: int = 0
    end_pos: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "confidence": self.confidence,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos
        }

@dataclass
class ChunkWithMetadata:
    """Chunk with associated metadata"""
    text: str
    doc_id: str
    chunk_id: str
    entities: List[Entity]
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

class EfficientNLPProcessor:
    """
    LLM-based NLP processor using OpenAI-compatible API for NER and coreference resolution.
    Uses gemma3:1b model for efficient processing.
    """
    
    def __init__(self):
        # NER LLM client
        self.ner_model = os.getenv("NER_MODEL", "gemma3:1b")
        self.ner_base_url = os.getenv("NER_BASE_URL", "http://localhost:11434/v1")
        ner_api_key_raw = os.getenv("NER_API_KEY", "test")
        self.ner_api_key = ner_api_key_raw.strip('"').strip("'") if ner_api_key_raw else "test"
        self.ner_temperature = float(os.getenv("NER_TEMPERATURE", "0.0"))
        
        # Coreference LLM client
        self.coref_model = os.getenv("COREF_MODEL", "gemma3:1b")
        self.coref_base_url = os.getenv("COREF_BASE_URL", "http://localhost:11434/v1")
        coref_api_key_raw = os.getenv("COREF_API_KEY", "test")
        self.coref_api_key = coref_api_key_raw.strip('"').strip("'") if coref_api_key_raw else "test"
        self.coref_temperature = float(os.getenv("COREF_TEMPERATURE", "0.0"))
        
        self.timeout = int(os.getenv("API_TIMEOUT", "600"))
        logger.info(f"Initialized LLM-based NLP processor with NER model: {self.ner_model}, Coref model: {self.coref_model}")
    
    async def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities using LLM-based NER with gemma3:1b model
        """
        if not text or not isinstance(text, str) or not text.strip():
            return []
        
        try:
            # Use the NER prompt from environment or default
            ner_prompt = os.getenv("NER_EXTRACTION_PROMPT", "")
            system_prompt = os.getenv("NER_SYSTEM_PROMPT", "You are a professional NER extraction assistant who responds only in valid python list of dictionaries.")
            
            user_prompt = f"{ner_prompt}{text}"
            
            # Call LLM for entity extraction
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await self._call_llm(
                messages, 
                self.ner_base_url, 
                self.ner_model, 
                self.ner_api_key, 
                self.ner_temperature
            )
            
            # Parse LLM response
            entities = self._parse_ner_response(response, text)
            
            logger.debug(f"Extracted {len(entities)} unique entities from text using LLM")
            return entities
            
        except Exception as e:
            logger.error(f"Error in LLM-based entity extraction: {e}")
            return []
    
    async def _call_llm(self, messages: List[Dict[str, str]], base_url: str, model: str, api_key: str, temperature: float) -> str:
        """Call OpenAI-compatible LLM API"""
        try:
            # Construct API URL - handle base_url that may already include /v1
            base_url = base_url.rstrip('/')
            if base_url.endswith('/v1'):
                api_url = f"{base_url}/chat/completions"
            else:
                api_url = f"{base_url}/v1/chat/completions"
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2048
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            if api_key and api_key != "test":
                headers["Authorization"] = f"Bearer {api_key}"
            
            async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
                response = await client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                # Extract content from response
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    raise ValueError("Invalid response format from LLM API")
                    
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _parse_ner_response(self, response: str, original_text: str) -> List[Entity]:
        """Parse LLM NER response into Entity objects with robust error handling"""
        entities = []
        original_response = response
        
        try:
            # Step 1: Clean up the response
            # Remove markdown code blocks if present
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```python\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            response = response.strip()
            
            # Remove leading/trailing text that might be before/after JSON
            # Try to find the JSON array more precisely
            json_patterns = [
                r'\[[\s\S]*?\]',  # Non-greedy match for array
                r'\{[\s\S]*?\}',  # Try object first
            ]
            
            entities_data = None
            
            # Strategy 1: Try to find and parse JSON array directly
            for pattern in json_patterns:
                json_match = re.search(pattern, response, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        # Try to fix common JSON issues
                        json_str = self._fix_json_string(json_str)
                        entities_data = json.loads(json_str)
                        break
                    except json.JSONDecodeError:
                        continue
            
            # Strategy 2: Try parsing entire response as JSON
            if entities_data is None:
                try:
                    cleaned = self._fix_json_string(response)
                    entities_data = json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
            
            # Strategy 3: Try to extract entities from text format
            if entities_data is None:
                entities_data = self._parse_text_format_entities(response)
            
            # Convert to list if needed
            if entities_data is None:
                logger.warning("Could not parse NER response in any format")
                logger.debug(f"Response was: {original_response[:500]}")
                return []
            
            if not isinstance(entities_data, list):
                logger.warning("LLM response is not a list, attempting to convert")
                entities_data = [entities_data] if isinstance(entities_data, dict) else []
            
            # Convert to Entity objects
            for entity_data in entities_data:
                if isinstance(entity_data, dict):
                    name = entity_data.get("name", "").strip()
                    entity_type = entity_data.get("type", "UNKNOWN").strip()
                    
                    if name and self._is_valid_entity(name, entity_type):
                        # Find position in original text
                        start_pos = original_text.lower().find(name.lower())
                        end_pos = start_pos + len(name) if start_pos >= 0 else 0
                        
                        entities.append(Entity(
                            name=name,
                            type=entity_type.upper(),
                            confidence=0.9,  # LLM-based, assume high confidence
                            start_pos=start_pos,
                            end_pos=end_pos
                        ))
            
            # Remove duplicates
            seen = set()
            unique_entities = []
            for entity in entities:
                key = (entity.name.lower(), entity.type)
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(entity)
            
            return unique_entities
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse NER response as JSON: {e}")
            logger.debug(f"Response was: {original_response[:500]}")
            return []
        except Exception as e:
            logger.error(f"Error parsing NER response: {e}")
            logger.debug(f"Response was: {original_response[:500]}")
            return []
    
    def _fix_json_string(self, json_str: str) -> str:
        """Try to fix common JSON formatting issues"""
        # Remove trailing commas before closing brackets/braces
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Fix single quotes to double quotes (basic cases)
        # This is tricky, so we'll be conservative
        json_str = re.sub(r"'(\w+)':", r'"\1":', json_str)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
        
        # Remove comments (JSON doesn't support comments)
        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        return json_str.strip()
    
    def _parse_text_format_entities(self, text: str) -> Optional[List[Dict[str, str]]]:
        """Fallback: Try to parse entities from text format like 'name: type'"""
        entities = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try patterns like: "name: type" or "{'name': '...', 'type': '...'}"
            # Pattern 1: name: type
            match = re.match(r'["\']?([^"\':]+)["\']?\s*[:=]\s*["\']?([^"\']+)["\']?', line)
            if match:
                name = match.group(1).strip()
                entity_type = match.group(2).strip()
                if name and entity_type:
                    entities.append({"name": name, "type": entity_type})
                    continue
            
            # Pattern 2: Try to extract from dict-like strings
            name_match = re.search(r'["\']name["\']\s*:\s*["\']([^"\']+)["\']', line, re.IGNORECASE)
            type_match = re.search(r'["\']type["\']\s*:\s*["\']([^"\']+)["\']', line, re.IGNORECASE)
            if name_match and type_match:
                entities.append({
                    "name": name_match.group(1).strip(),
                    "type": type_match.group(1).strip()
                })
        
        return entities if entities else None
    
    def _is_valid_entity(self, text: str, entity_type: str) -> bool:
        """Filter out invalid entities"""
        text = text.strip()
        
        # Basic filters
        if (len(text) < 2 or len(text) > 100 or 
            text.isdigit() or 
            re.match(r'^[^\w\s]+$', text) or
            text.lower() in {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}):
            return False

        return True

    async def resolve_coreferences(self, text: str) -> str:
        """
        Resolve coreferences using LLM-based approach with gemma3:1b model
        """
        if not text or not text.strip():
            return text
        
        try:
            # Use the coreference prompt from environment or default
            coref_user_prompt = os.getenv("COREF_USER_PROMPT", "Resolve all ambiguous pronouns and vague references in the following text by replacing them with their appropriate entities. Ensure no unresolved pronouns like 'he', 'she', 'himself', etc. remain. If the referenced entity is unclear, make an intelligent guess based on context.\n\n")
            coref_system_prompt = os.getenv("COREF_SYSTEM_PROMPT", "You are a professional text refiner skilled in coreference resolution.")
            
            user_prompt = f"{coref_user_prompt}{text}"
            
            messages = [
                {"role": "system", "content": coref_system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            resolved_text = await self._call_llm(
                messages,
                self.coref_base_url,
                self.coref_model,
                self.coref_api_key,
                self.coref_temperature
            )
            
            logger.debug("Coreference resolution completed using LLM")
            return resolved_text
            
        except Exception as e:
            logger.warning(f"LLM-based coreference resolution failed: {e}, returning original text")
            return text

class AsyncLLMClient:
    """
    Async LLM client with improved error handling and retry logic
    """
    
    def __init__(self):
        # Strip quotes if present (common when values are quoted in .env files)
        api_key_raw = os.getenv("OPENAI_API_KEY")
        self.api_key = api_key_raw.strip('"').strip("'") if api_key_raw else None
        self.model = os.getenv("OPENAI_MODEL", "llama3.2")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
        self.timeout = int(os.getenv("API_TIMEOUT", "600"))
        self.provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        
        stop_str = os.getenv("OPENAI_STOP")
        self.stop = json.loads(stop_str) if stop_str else None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
    
    async def invoke(self, messages: List[Dict[str, str]], max_retries: int = 3) -> str:
        """Invoke LLM with improved error handling and rate limiting"""
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        if self.provider in {"openai", "ollama"}:
            return await self._invoke_openai(messages, max_retries)
        else:
            return await self._invoke_gemini(messages, max_retries)

    async def _invoke_gemini(self, messages: List[Dict[str, str]], max_retries: int) -> str:
        """Call Gemini-compatible endpoint."""
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                # Gemini expects user/model; fold system into the first user message
                if contents:
                    contents[0]["parts"][0]["text"] = f"{msg.get('content', '')}\n\n{contents[0]['parts'][0]['text']}"
                else:
                    contents.append({"role": "user", "parts": [{"text": msg.get("content", "")}]})
            else:
                contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 2048
            }
        }

        headers = {
            "Content-Type": "application/json"
        }

        api_url = f"{self.base_url}/models/{self.model}:generateContent"

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
                    response = await client.post(
                        api_url,
                        json=payload,
                        headers=headers,
                        params={"key": self.api_key} if self.api_key else None
                    )
                    response.raise_for_status()

                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates and candidates[0].get("content", {}).get("parts"):
                        content = candidates[0]["content"]["parts"][0].get("text", "").strip()
                        if not content:
                            raise ValueError("Empty response from LLM")

                        self.last_request_time = time.time()
                        return content
                    else:
                        raise ValueError("Invalid response format from Gemini API")

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
                if e.response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    logger.info(f"Rate limited, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                elif attempt == max_retries - 1:
                    raise
                else:
                    await asyncio.sleep(1 * (attempt + 1))

            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))

        raise Exception("Max retries exceeded")

    async def _invoke_openai(self, messages: List[Dict[str, str]], max_retries: int) -> str:
        """Call OpenAI/Ollama-compatible chat completions endpoint."""
        # Construct URL - handle base_url that may already include /v1
        base_url = self.base_url.rstrip('/')
        if base_url.endswith('/v1'):
            url = f"{base_url}/chat/completions"
        else:
            url = f"{base_url}/v1/chat/completions"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.stop:
            payload["stop"] = self.stop

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()

                    data = response.json()
                    choices = data.get("choices", [])
                    if choices:
                        message = choices[0].get("message", {})
                        content = (message.get("content") or "").strip()
                        if not content:
                            raise ValueError("Empty response from LLM")

                        self.last_request_time = time.time()
                        return content
                    else:
                        raise ValueError("Invalid response format from OpenAI-compatible API")

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
                if e.response.status_code == 429:
                    wait_time = 2 ** attempt
                    logger.info(f"Rate limited, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                elif attempt == max_retries - 1:
                    raise
                else:
                    await asyncio.sleep(1 * (attempt + 1))

            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))

        raise Exception("Max retries exceeded")

class BatchEmbeddingClient:
    """
    Efficient batch embedding client with caching and optimization
    Supports both Ollama/OpenAI and Gemini embedding APIs
    """
    
    def __init__(self):
        self.embedding_api_url = os.getenv("EMBEDDING_API_URL", "http://localhost/api/embed")
        # Strip quotes if present (common when values are quoted in .env files)
        embedding_api_key_raw = os.getenv("EMBEDDING_API_KEY")
        self.embedding_api_key = embedding_api_key_raw.strip('"').strip("'") if embedding_api_key_raw else None
        self.model_name = os.getenv("EMBEDDING_MODEL_NAME", "mxbai-embed-large")
        self.timeout = int(os.getenv("API_TIMEOUT", "600"))
        self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.include_bearer_auth = os.getenv("EMBEDDING_INCLUDE_BEARER_AUTH", "true").lower() == "true"
        
        # In-memory cache with TTL
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = CACHE_TTL
        
        logger.info(f"Initialized BatchEmbeddingClient with provider={self.provider}, batch_size={self.batch_size}")
        if not self.embedding_api_key:
            logger.warning("EMBEDDING_API_KEY not found in environment variables")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for caching"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _is_cache_valid(self, text_hash: str) -> bool:
        """Check if cached embedding is still valid"""
        if text_hash not in self.cache:
            return False
        
        timestamp = self.cache_timestamps.get(text_hash, 0)
        return (time.time() - timestamp) < self.cache_ttl
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get single embedding with caching"""
        if not text or not text.strip():
            raise ValueError("Empty text provided for embedding")
        
        text_hash = self._get_text_hash(text)
        
        # Check cache
        if self._is_cache_valid(text_hash):
            logger.debug(f"Embedding cache hit for hash: {text_hash}")
            return self.cache[text_hash]
        
        # Get embedding via batch method
        embeddings = await self.get_embeddings([text])
        return embeddings[0] if embeddings else []
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get multiple embeddings efficiently with batching"""
        if not texts:
            return []
        
        # Filter out empty texts and check cache
        valid_texts = []
        cached_embeddings = {}
        text_to_index = {}
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
                
            text_hash = self._get_text_hash(text)
            text_to_index[text] = i
            
            if self._is_cache_valid(text_hash):
                cached_embeddings[i] = self.cache[text_hash]
                logger.debug(f"Cache hit for text {i}")
            else:
                valid_texts.append((i, text))
        
        # Process uncached texts in batches
        results = [None] * len(texts)
        
        # Fill in cached results
        for idx, embedding in cached_embeddings.items():
            results[idx] = embedding
        
        # Process remaining texts in batches
        if valid_texts:
            for i in range(0, len(valid_texts), self.batch_size):
                batch = valid_texts[i:i + self.batch_size]
                batch_texts = [text for _, text in batch]
                batch_indices = [idx for idx, _ in batch]
                
                try:
                    # Choose embedding method based on provider
                    if self.provider in {"openai", "ollama"}:
                        batch_embeddings = await self._fetch_batch_embeddings_openai(batch_texts)
                    else:  # gemini
                        batch_embeddings = await self._fetch_batch_embeddings_gemini(batch_texts)
                    
                    # Store results and update cache
                    for j, embedding in enumerate(batch_embeddings):
                        original_idx = batch_indices[j]
                        results[original_idx] = embedding
                        
                        # Update cache
                        text_hash = self._get_text_hash(batch_texts[j])
                        self.cache[text_hash] = embedding
                        self.cache_timestamps[text_hash] = time.time()
                        
                except Exception as e:
                    logger.error(f"Batch embedding failed: {e}")
                    # Fill with empty embeddings for failed batch
                    for idx in batch_indices:
                        results[idx] = []
        
        # Clean up None values
        return [emb if emb is not None else [] for emb in results]
    
    async def _fetch_batch_embeddings_openai(self, texts: List[str]) -> List[List[float]]:
        """Fetch embeddings for a batch of texts using OpenAI/Ollama compatible API"""
        # OpenAI/Ollama API format for embeddings
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        # Prepare headers for OpenAI/Ollama API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.embedding_api_key}"
        }
        
        # Ensure OpenAI/Ollama URL ends with /embeddings
        api_url = self.embedding_api_url.rstrip('/')
        if not api_url.endswith('/embeddings'):
            api_url = f"{api_url}/embeddings"
        
        async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
            response = await client.post(
                api_url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            embeddings_data = data.get("data", [])
            
            # Extract the actual embedding values
            embeddings = []
            for emb_data in embeddings_data:
                embedding = emb_data.get("embedding", [])
                embeddings.append(embedding)
            
            if len(embeddings) != len(texts):
                raise ValueError(f"Invalid embedding response: expected {len(texts)}, got {len(embeddings)}")
            
            return embeddings
    
    async def _fetch_batch_embeddings_gemini(self, texts: List[str]) -> List[List[float]]:
        """Fetch embeddings for a batch of texts using Gemini API"""
        # Gemini API format for embeddings
        requests = []
        for text in texts:
            requests.append({
                "model": f"models/{self.model_name}",
                "content": {
                    "parts": [{"text": text}]
                }
            })

        payload = {"requests": requests}

        # Prepare headers for Gemini API
        # Gemini API uses x-goog-api-key header, not Authorization Bearer
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.embedding_api_key
        }

        # Construct the correct Gemini API URL for batch embeddings
        # For Gemini, remove /embeddings from URL if present, as it needs base URL + /models/...
        base_url = self.embedding_api_url.rstrip('/')
        if '/embeddings' in base_url:
            base_url = base_url.split('/embeddings')[0]  # Remove '/embeddings' and anything after
        api_url = f"{base_url}/models/{self.model_name}:batchEmbedContents"
        
        async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
            response = await client.post(
                api_url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            embeddings_data = data.get("embeddings", [])
            
            # Extract the actual embedding values
            embeddings = []
            for emb_data in embeddings_data:
                values = emb_data.get("values", [])
                embeddings.append(values)
            
            if len(embeddings) != len(texts):
                raise ValueError(f"Invalid embedding response: expected {len(texts)}, got {len(embeddings)}")
            
            return embeddings

class ImprovedTextProcessor:
    """
    Improved text processing with better chunking and cleaning
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Enhanced text cleaning with boilerplate and navigation text removal"""
        if not isinstance(text, str):
            logger.error(f"Received non-string content: {type(text)}")
            return ""
        
        # Remove non-printable characters
        text = ''.join(filter(lambda x: x in string.printable, text.strip()))
        
        # Remove common boilerplate/navigation patterns
        boilerplate_patterns = [
            r'\b(page|Page|PAGE)\s+\d+\s+of\s+\d+\b',  # Page numbers
            r'\b(page|Page|PAGE)\s+\d+\b',  # Simple page numbers
            r'^\s*Table of Contents.*?\n',  # Table of contents
            r'^\s*Contents.*?\n',
            r'^\s*Index.*?\n',
            r'^\s*Appendix.*?\n',
            r'^\s*Chapter\s+\d+.*?\n',
            r'^\s*Section\s+\d+.*?\n',
            r'^\s*\d+\.\s*\d+\.\s*\d+.*?\n',  # Numbered sections
            r'^\s*\[.*?\]\s*$',  # Bracketed navigation text
            r'^\s*←\s*.*?\s*→\s*$',  # Navigation arrows
            r'^\s*Previous\s+Page.*?Next\s+Page\s*$',
            r'^\s*Back\s+to\s+Top\s*$',
            r'^\s*Home\s*$',
            r'^\s*Menu\s*$',
            r'^\s*Navigation\s*$',
            r'^\s*Skip to content\s*$',
            r'^\s*Copyright\s+©.*?$',  # Copyright notices
            r'^\s*All rights reserved.*?$',
            r'^\s*Confidential.*?$',
            r'^\s*DRAFT.*?$',
            r'^\s*PROPRIETARY.*?$',
            r'^\s*Header:.*?$',  # Headers/footers
            r'^\s*Footer:.*?$',
            r'^\s*Header\s+\d+.*?$',
            r'^\s*Footer\s+\d+.*?$',
            r'^\s*\[.*?Header.*?\]\s*$',
            r'^\s*\[.*?Footer.*?\]\s*$',
            r'^\s*Click here.*?$',  # Interactive elements
            r'^\s*See also.*?$',
            r'^\s*Related links.*?$',
            r'^\s*Bookmark.*?$',
            r'^\s*Print.*?$',
            r'^\s*Share.*?$',
            r'^\s*Download.*?$',
            r'^\s*Email.*?$',
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove URLs (often navigation)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses (often in headers/footers)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove repeated navigation-like phrases
        navigation_phrases = [
            r'\b(?:previous|next|back|forward|up|down)\s+(?:page|section|chapter)\b',
            r'\b(?:go to|jump to|see|refer to)\s+(?:page|section|chapter)\s+\d+\b',
        ]
        for phrase in navigation_phrases:
            text = re.sub(phrase, '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common document artifacts
        text = re.sub(r'\n\s*\n', '\n', text)  # Multiple newlines
        text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)  # Bullet points
        
        # Remove lines that are too short and likely navigation (less than 10 chars, mostly punctuation)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 10 or (len(line) > 0 and not re.match(r'^[^\w\s]+$', line)):
                cleaned_lines.append(line)
        text = '\n'.join(cleaned_lines)
        
        return text.strip()
    
    @staticmethod
    def chunk_text_advanced(text: str, max_chunk_size: int = CHUNK_SIZE_GDS, overlap: int = 50) -> List[str]:
        """
        Advanced text chunking with overlap and sentence boundary preservation
        """
        if not text or not text.strip():
            return []
        
        # Use blingfire for sentence splitting
        sentences = [s.strip() for s in blingfire.text_to_sentences(text).splitlines() if s.strip()]
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed the limit
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + " " + sentence
                    current_size = len(current_chunk)
                else:
                    current_chunk = sentence
                    current_size = sentence_size
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                    current_size += sentence_size + 1
                else:
                    current_chunk = sentence
                    current_size = sentence_size
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        logger.debug(f"Split text into {len(chunks)} chunks with overlap={overlap}")
        return chunks

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Optimized cosine similarity calculation"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    try:
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0

async def run_cypher_query_async(driver, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Async wrapper for Neo4j queries to avoid blocking the event loop
    """
    parameters = parameters or {}
    
    def _run_query():
        try:
            with driver.session() as session:
                result = session.run(query, **parameters)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Neo4j query error: {e}")
            raise HTTPException(status_code=500, detail=f"Database query failed: {e}")
    
    # Run in thread pool to avoid blocking
    return await asyncio.get_event_loop().run_in_executor(thread_pool, _run_query)

def run_async_safe(coro):
    """
    Safely run async coroutine even if event loop is already running
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        # Create new thread with new event loop
        result_container = {}
        
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                result_container["result"] = new_loop.run_until_complete(coro)
            except Exception as e:
                result_container["result"] = e
            finally:
                new_loop.close()
        
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
        
        result = result_container.get("result")
        if isinstance(result, Exception):
            raise result
        return result
    else:
        return asyncio.run(coro)

# Global instances
nlp_processor = EfficientNLPProcessor()
llm_client = AsyncLLMClient()
embedding_client = BatchEmbeddingClient()
text_processor = ImprovedTextProcessor()

# Convenience functions for backward compatibility
async def extract_entities_efficient(text: str) -> List[Dict[str, str]]:
    """Extract entities using LLM-based approach with gemma3:1b"""
    entities = await nlp_processor.extract_entities(text)
    return [entity.to_dict() for entity in entities]

async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings in batches for efficiency"""
    return await embedding_client.get_embeddings(texts)

def clean_text_improved(text: str) -> str:
    """Clean text using improved processor"""
    return text_processor.clean_text(text)

def chunk_text_improved(text: str, max_chunk_size: int = CHUNK_SIZE_GDS) -> List[str]:
    """Chunk text using improved processor"""
    return text_processor.chunk_text_advanced(text, max_chunk_size)
