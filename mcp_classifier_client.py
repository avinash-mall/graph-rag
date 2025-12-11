"""
MCP Client for Question Classification

This client connects to the MCP classification server and calls the classify_question tool.

Features:
- Resilience patterns with circuit breakers and retries
- Structured logging with context fields
- Graceful fallback on connection errors

This module uses:
- config.py: Centralized configuration (MCP URL, timeout settings)
- logging_config.py: Standardized structured logging with context fields
- resilience.py: Automatic retries and circuit breaking for MCP calls
"""

import asyncio
from typing import Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from question_classifier import ClassificationResult, QuestionType

# Import centralized configuration, logging, and resilience
from config import get_config
from logging_config import get_logger, log_external_service_call, log_error_with_context
from resilience import call_with_resilience, get_circuit_breaker

# Get configuration
cfg = get_config()

# Setup logging using centralized logging config
logger = get_logger("MCPClassifierClient")

# Configuration from centralized config
MCP_CLASSIFIER_URL = cfg.classifier.mcp_config.url
MCP_CLASSIFIER_TIMEOUT = cfg.classifier.mcp_config.timeout

# Global session cache
_client_session: Optional[ClientSession] = None
_read_stream = None
_write_stream = None

async def classify_question_via_mcp(question: str) -> ClassificationResult:
    """
    Classify a question using the MCP classification server with resilience.
    
    Args:
        question: User's question to classify
        
    Returns:
        ClassificationResult with type, reason, and confidence
    """
    # Get circuit breaker for MCP service
    mcp_circuit_breaker = get_circuit_breaker("mcp_classifier")
    
    # Create context for logging
    context = log_external_service_call(
        service="mcp_classifier",
        endpoint="classify_question",
        method="POST",
        question_preview=question[:100]
    )
    
    async def _classify():
        # Connect to the MCP server
        async with streamablehttp_client(MCP_CLASSIFIER_URL) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize the MCP session
                await session.initialize()
                
                # Call the classify_question tool
                result = await session.call_tool(
                    "classify_question",
                    arguments={"question": question},
                )
                
                # Extract structured content (thanks to json_response=True on server)
                structured_result = result.structuredContent
                
                if structured_result and isinstance(structured_result, dict):
                    # Validate the result
                    question_type = structured_result.get("type", "").upper()
                    if question_type in ["BROAD", "CHUNK", "OUT_OF_SCOPE"]:
                        return {
                            "type": question_type,
                            "reason": structured_result.get("reason", "Classified via MCP"),
                            "confidence": float(structured_result.get("confidence", 0.5))
                        }
                    else:
                        logger.warning(
                            f"Invalid classification type from MCP: {question_type}",
                            extra=context
                        )
                        # Fallback
                        return {
                            "type": "CHUNK",
                            "reason": "Invalid MCP response, defaulting to CHUNK",
                            "confidence": 0.5
                        }
                    else:
                        logger.warning(
                            f"Invalid MCP response structure: {structured_result}",
                            extra=context
                        )
                        # Fallback
                        return {
                            "type": "CHUNK",
                            "reason": "Invalid MCP response, defaulting to CHUNK",
                            "confidence": 0.5
                        }
    
    try:
        # Use resilience for MCP call
        return await call_with_resilience(
            _classify,
            circuit_breaker=mcp_circuit_breaker,
            timeout=MCP_CLASSIFIER_TIMEOUT,
            context=context
        )
    except Exception as e:
        log_error_with_context(
            logger,
            f"Error calling MCP classification server: {e}",
            exception=e,
            context=context
        )
        # Fallback to basic classification
        return {
            "type": "CHUNK",
            "reason": f"MCP classification failed: {str(e)}, defaulting to CHUNK",
            "confidence": 0.5
        }

async def test_mcp_connection() -> bool:
    """
    Test if the MCP server is reachable and working.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        async with streamablehttp_client(MCP_CLASSIFIER_URL) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                # List tools to verify server is working
                tools_response = await session.list_tools()
                tool_names = [tool.name for tool in tools_response.tools]
                
                if "classify_question" in tool_names:
                    logger.info(f"MCP server connection successful. Available tools: {tool_names}")
                    return True
                else:
                    logger.warning(f"MCP server connected but 'classify_question' tool not found. Tools: {tool_names}")
                    return False
                    
    except Exception as e:
        logger.error(f"MCP server connection test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the MCP client
    async def main():
        print("Testing MCP classifier connection...")
        connected = await test_mcp_connection()
        
        if connected:
            print("\nTesting classification...")
            test_questions = [
                "Give me an overview of all policies",
                "What is the deadline mentioned in document X?",
                "Tell me about the main themes in the documents"
            ]
            
            for question in test_questions:
                result = await classify_question_via_mcp(question)
                print(f"\nQuestion: {question}")
                print(f"Classification: {result['type']}")
                print(f"Reason: {result['reason']}")
                print(f"Confidence: {result['confidence']:.2f}")
        else:
            print("Failed to connect to MCP server. Make sure it's running.")
    
    asyncio.run(main())

