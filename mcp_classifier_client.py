"""
MCP Client for Question Classification

This client connects to the MCP classification server and calls the classify_question tool.
"""

import asyncio
import logging
import os
from typing import Optional
from dotenv import load_dotenv

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from question_classifier import ClassificationResult, QuestionType

load_dotenv()

logger = logging.getLogger("MCPClassifierClient")

# Configuration
MCP_CLASSIFIER_URL = os.getenv("MCP_CLASSIFIER_URL", "http://localhost:8001/mcp")
MCP_CLASSIFIER_TIMEOUT = int(os.getenv("MCP_CLASSIFIER_TIMEOUT", "30"))

# Global session cache
_client_session: Optional[ClientSession] = None
_read_stream = None
_write_stream = None

async def classify_question_via_mcp(question: str) -> ClassificationResult:
    """
    Classify a question using the MCP classification server.
    
    Args:
        question: User's question to classify
        
    Returns:
        ClassificationResult with type, reason, and confidence
    """
    try:
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
                        logger.warning(f"Invalid classification type from MCP: {question_type}")
                        # Fallback
                        return {
                            "type": "CHUNK",
                            "reason": "Invalid MCP response, defaulting to CHUNK",
                            "confidence": 0.5
                        }
                else:
                    logger.warning(f"Invalid MCP response structure: {structured_result}")
                    # Fallback
                    return {
                        "type": "CHUNK",
                        "reason": "Invalid MCP response, defaulting to CHUNK",
                        "confidence": 0.5
                    }
                    
    except Exception as e:
        logger.error(f"Error calling MCP classification server: {e}")
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

