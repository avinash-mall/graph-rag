"""
Utilities for Graph RAG with efficient NLP preprocessing and optimized performance.

Features:
- Fast spaCy-based NLP processing (200-500x faster than LLM-based NER)
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
import spacy
import blingfire
from transformers import pipeline
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
    Efficient NLP processor using spaCy for NER and other NLP tasks.
    Replaces the expensive LLM-based approach with fast, local models.
    """
    
    def __init__(self):
        self.nlp = None
        self.coref_pipeline = None
        self._load_models()
    
    def _load_models(self):
        """Load spaCy and other NLP models"""
        try:
            # Try to load the large English model first
            try:
                self.nlp = spacy.load("en_core_web_lg")
                logger.info("Loaded spaCy large model (en_core_web_lg)")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded spaCy small model (en_core_web_sm)")
                except OSError:
                    logger.error("No spaCy English model found. Please install with: python -m spacy download en_core_web_sm")
                    raise
            
            # Use simple coreference resolution to avoid large model downloads
            # This is more efficient and sufficient for most use cases
            self.coref_pipeline = None
            logger.info("Using simple fallback coreference resolution")
                
        except Exception as e:
            logger.error(f"Failed to load NLP models: {e}")
            raise
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities using spaCy NER - much faster than LLM-based approach
        """
        if not text or not isinstance(text, str) or not text.strip():
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                # Map spaCy entity types to our standardized types
                entity_type = self._map_entity_type(ent.label_)
                
                # Filter out low-confidence or irrelevant entities
                if self._is_valid_entity(ent.text, entity_type):
                    entities.append(Entity(
                        name=ent.text.strip(),
                        type=entity_type,
                        confidence=1.0,  # spaCy doesn't provide confidence scores by default
                        start_pos=ent.start_char,
                        end_pos=ent.end_char
                    ))
            
            # Remove duplicates while preserving order
            seen = set()
            unique_entities = []
            for entity in entities:
                key = (entity.name.lower(), entity.type)
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(entity)
            
            logger.debug(f"Extracted {len(unique_entities)} unique entities from text")
            return unique_entities
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return []
    
    def _map_entity_type(self, spacy_label: str) -> str:
        """Map spaCy entity labels to our standardized types"""
        mapping = {
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION", 
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "PRODUCT": "PRODUCT",
            "EVENT": "EVENT",
            "WORK_OF_ART": "PRODUCT",
            "LAW": "CONCEPT",
            "LANGUAGE": "CONCEPT",
            "DATE": "DATE",
            "TIME": "DATE",
            "PERCENT": "METRIC",
            "MONEY": "METRIC",
            "QUANTITY": "METRIC",
            "ORDINAL": "METRIC",
            "CARDINAL": "METRIC",
            "FAC": "LOCATION",
            "NORP": "ORGANIZATION"
        }
        return mapping.get(spacy_label, "CONCEPT")
    
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
    
    def resolve_coreferences(self, text: str) -> str:
        """
        Resolve coreferences using a lightweight approach.
        Falls back to simple pronoun replacement if advanced model unavailable.
        """
        if not text or not text.strip():
            return text
        
        try:
            if self.coref_pipeline:
                # Use T5-based coreference resolution with proper prompt
                prompt = f"Resolve pronouns in this text: {text}"
                result = self.coref_pipeline(prompt, max_length=512, truncation=True)
                return result[0]['generated_text'] if result else text
            else:
                # Simple pronoun replacement as fallback
                return self._simple_coref_resolution(text)
                
        except Exception as e:
            logger.warning(f"Coreference resolution failed: {e}")
            return text
    
    def _simple_coref_resolution(self, text: str) -> str:
        """Simple rule-based coreference resolution"""
        doc = self.nlp(text)
        
        # Extract person names
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        
        if persons:
            # Replace pronouns with the most recent person mentioned
            most_recent_person = persons[-1]
            
            # Simple pronoun replacement
            text = re.sub(r'\bhe\b', most_recent_person, text, flags=re.IGNORECASE)
            text = re.sub(r'\bhim\b', most_recent_person, text, flags=re.IGNORECASE)
            text = re.sub(r'\bhis\b', f"{most_recent_person}'s", text, flags=re.IGNORECASE)
        
        return text

class AsyncLLMClient:
    """
    Async LLM client with improved error handling and retry logic
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "llama3.2")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
        self.timeout = int(os.getenv("API_TIMEOUT", "600"))
        
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
        
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            if msg.get("role") == "user":
                contents.append({"parts": [{"text": msg.get("content", "")}]})
            elif msg.get("role") == "system":
                # Gemini doesn't have system role, prepend to first user message
                if contents:
                    contents[0]["parts"][0]["text"] = f"{msg.get('content', '')}\n\n{contents[0]['parts'][0]['text']}"
                else:
                    contents.append({"parts": [{"text": msg.get("content", "")}]})
        
        # Gemini API format
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 2048
            }
        }
        
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Construct the correct Gemini API URL
        api_url = f"{self.base_url}/models/{self.model}:generateContent"
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
                    response = await client.post(api_url, json=payload, headers=headers)
                    response.raise_for_status()
                    
                    data = response.json()
                    # Gemini API response format
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

class BatchEmbeddingClient:
    """
    Efficient batch embedding client with caching and optimization
    """
    
    def __init__(self):
        self.embedding_api_url = os.getenv("EMBEDDING_API_URL", "http://localhost/api/embed")
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY")
        self.model_name = os.getenv("EMBEDDING_MODEL_NAME", "mxbai-embed-large")
        self.timeout = int(os.getenv("API_TIMEOUT", "600"))
        self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
        
        # In-memory cache with TTL
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = CACHE_TTL
        
        logger.info(f"Initialized BatchEmbeddingClient with batch_size={self.batch_size}")
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
                    batch_embeddings = await self._fetch_batch_embeddings(batch_texts)
                    
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
    
    async def _fetch_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
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
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.embedding_api_key
        }
        
        # Construct the correct Gemini API URL for batch embeddings
        api_url = f"{self.embedding_api_url}/models/{self.model_name}:batchEmbedContents"
        
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
        """Enhanced text cleaning"""
        if not isinstance(text, str):
            logger.error(f"Received non-string content: {type(text)}")
            return ""
        
        # Remove non-printable characters
        text = ''.join(filter(lambda x: x in string.printable, text.strip()))
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common document artifacts
        text = re.sub(r'\n\s*\n', '\n', text)  # Multiple newlines
        text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)  # Bullet points
        text = re.sub(r'\b(page|Page)\s+\d+\b', '', text)  # Page numbers
        
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
    """Extract entities using efficient spaCy-based approach"""
    entities = nlp_processor.extract_entities(text)
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
