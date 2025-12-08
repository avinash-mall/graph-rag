"""
MCP Server for Question Classification

This server exposes a classify_question tool that can be called via MCP protocol.
The classification logic is agentic and uses LLM with fallback heuristics.
"""

import logging
import os
import json
import re
from typing import Literal, TypedDict
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Set PORT environment variable before importing FastMCP
# FastMCP's streamable-http transport reads PORT from environment
server_port = os.getenv("MCP_CLASSIFIER_PORT", "8001")
os.environ["PORT"] = server_port

from mcp.server.fastmcp import FastMCP
from utils import llm_client

logger = logging.getLogger("MCPClassifierServer")

QuestionType = Literal["BROAD", "CHUNK", "OUT_OF_SCOPE"]

class ClassificationResult(TypedDict):
    """Result of question classification"""
    type: QuestionType
    reason: str
    confidence: float

# Create the MCP server
mcp = FastMCP(
    name="question-classifier",
    json_response=True,      # Nice JSON responses for HTTP clients
    stateless_http=True,     # Good default for HTTP transport
)

# Configuration
use_heuristics = os.getenv("CLASSIFIER_USE_HEURISTICS", "true").lower() == "true"
use_llm = os.getenv("CLASSIFIER_USE_LLM", "true").lower() == "true"

def _classify_with_heuristics(question: str) -> ClassificationResult:
    """
    Classify using simple heuristic rules.
    Fast but less accurate than LLM.
    """
    q_lower = question.lower()
    
    # Broad question indicators
    broad_indicators = [
        "overview", "high level", "summary", "main themes", "main topics",
        "general", "overall", "broadly", "in general", "all about",
        "what are the", "give me an overview", "tell me about",
        "compare", "differences between", "similarities",
        "trends", "patterns", "across all", "multiple", "various"
    ]
    
    broad_score = sum(1 for indicator in broad_indicators if indicator in q_lower)
    if broad_score >= 1:
        return {
            "type": "BROAD",
            "reason": f"Question asks for overview/broad understanding (matched {broad_score} indicators)",
            "confidence": min(0.9, 0.6 + broad_score * 0.1)
        }
    
    # Chunk-specific indicators
    chunk_indicators = [
        "detail", "exact", "according to", "section", "chapter",
        "paragraph", "quote", "where does it say", "which document",
        "when did", "what date", "who said", "specifically",
        "precise", "exactly", "mention", "stated"
    ]
    
    chunk_score = sum(1 for indicator in chunk_indicators if indicator in q_lower)
    if chunk_score >= 2:
        return {
            "type": "CHUNK",
            "reason": f"Question asks for specific details (matched {chunk_score} indicators)",
            "confidence": min(0.9, 0.7 + chunk_score * 0.1)
        }
    
    # Default to CHUNK for specific-looking questions
    if len(question.split()) <= 5:  # Short questions often specific
        return {
            "type": "CHUNK",
            "reason": "Short question likely requires specific answer",
            "confidence": 0.6
        }
    
    # Default fallback
    return {
        "type": "CHUNK",
        "reason": "Default classification to chunk-based retrieval",
        "confidence": 0.5
    }

async def _classify_with_llm(question: str) -> ClassificationResult | None:
    """
    Classify using LLM for more accurate classification.
    """
    try:
        classification_prompt = f"""You are a question classifier for a knowledge base system. Classify the user's question into EXACTLY one of these categories:

1. BROAD: Questions that need broad understanding, overviews, summaries across multiple documents or topics, comparisons, general trends
2. CHUNK: Questions that need specific details, exact quotes, precise information from specific document sections
3. OUT_OF_SCOPE: Questions that cannot be answered from a local knowledge base (e.g., real-time events, personal questions, unrelated topics)

Return ONLY a JSON object with these exact keys:
- "type": one of "BROAD", "CHUNK", or "OUT_OF_SCOPE"
- "reason": a brief explanation (1-2 sentences)
- "confidence": a number between 0.0 and 1.0

Question: "{question}"

JSON Response:"""

        messages = [
            {
                "role": "system",
                "content": "You are a precise JSON classifier. Respond only with valid JSON, no additional text."
            },
            {
                "role": "user",
                "content": classification_prompt
            }
        ]
        
        response = await llm_client.invoke(messages)
        
        # Parse JSON from response
        result = _parse_classification_response(response)
        
        if result and result.get("type") in ["BROAD", "CHUNK", "OUT_OF_SCOPE"]:
            return result
        else:
            logger.warning(f"Invalid LLM classification result: {result}")
            return None
                
    except Exception as e:
        logger.error(f"Error in LLM classification: {e}")
        return None

def _parse_classification_response(response: str) -> ClassificationResult | None:
    """Parse LLM response into ClassificationResult"""
    try:
        # Remove markdown code blocks if present
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        response = response.strip()
        
        # Try to find JSON object
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            result = json.loads(response)
        
        # Validate and normalize
        if not isinstance(result, dict):
            return None
        
        question_type = result.get("type", "").upper()
        if question_type not in ["BROAD", "CHUNK", "OUT_OF_SCOPE"]:
            return None
        
        confidence = float(result.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        
        return {
            "type": question_type,
            "reason": result.get("reason", "Classified by LLM"),
            "confidence": confidence
        }
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to parse classification response: {e}")
        return None

async def _classify_question_async(question: str) -> ClassificationResult:
    """
    Internal async classification logic.
    """
    if not question or not question.strip():
        return {
            "type": "OUT_OF_SCOPE",
            "reason": "Empty or invalid question",
            "confidence": 1.0
        }
    
    # Step 1: Try heuristics first (fast, cheap)
    if use_heuristics:
        heuristic_result = _classify_with_heuristics(question)
        if heuristic_result["confidence"] >= 0.8:
            logger.info(f"High-confidence heuristic classification: {heuristic_result['type']}")
            return heuristic_result
    
    # Step 2: Use LLM for classification (more accurate)
    if use_llm:
        try:
            llm_result = await _classify_with_llm(question)
            if llm_result:
                logger.info(f"LLM classification: {llm_result['type']} - {llm_result['reason']}")
                return llm_result
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, falling back to heuristics")
    
    # Step 3: Fallback to default heuristic result
    return _classify_with_heuristics(question)

# Use a simple sync wrapper that calls async internally
# FastMCP handles async tools properly when using streamable-http
@mcp.tool()
async def classify_question(question: str) -> ClassificationResult:
    """
    Classify a user question into BROAD, CHUNK, or OUT_OF_SCOPE.
    
    - BROAD: Questions needing broad understanding/overview → use community summaries
    - CHUNK: Questions needing specific details → use chunk-level retrieval
    - OUT_OF_SCOPE: Questions not covered by knowledge base
    
    Args:
        question: The user's question to classify
        
    Returns:
        ClassificationResult with type, reason, and confidence score
    """
    return await _classify_question_async(question)

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment
    server_port = int(os.getenv("MCP_CLASSIFIER_PORT", "8001"))
    os.environ["PORT"] = str(server_port)
    
    logger.info(f"Starting MCP classifier server on port {server_port}")
    logger.info(f"Server will be available at: http://0.0.0.0:{server_port}/mcp")
    
    # Use uvicorn directly with FastMCP's streamable_http_app
    # This allows us to explicitly set the port, which mcp.run() doesn't support
    try:
        # Get the ASGI app from FastMCP
        asgi_app = mcp.streamable_http_app
        
        # Run with uvicorn explicitly setting the port
        uvicorn.run(
            asgi_app,
            host="0.0.0.0",
            port=server_port,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Error running with uvicorn: {e}")
        logger.info("Falling back to mcp.run()...")
        # Fallback to default run method
        mcp.run(transport="streamable-http")

