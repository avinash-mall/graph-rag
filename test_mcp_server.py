"""
Test script for MCP Server functionality

Tests:
1. MCP server connection
2. Broad question (should use document summaries and map-reduce)
3. Specific question (should use chunks and relations)
"""

import asyncio
import json
import sys
import os
from typing import Dict, Any, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_classifier_client import classify_question_via_mcp, test_mcp_connection
from unified_search import get_search_pipeline, SearchScope, UnifiedSearchPipeline
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Configuration
DB_URL = os.getenv("DB_URL")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Global driver instance to be reused across tests
_global_driver: Optional[Any] = None

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def print_result(result: Dict[str, Any], indent: int = 0):
    """Pretty print a result dictionary"""
    prefix = "  " * indent
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_result(value, indent + 1)
        elif isinstance(value, list):
            print(f"{prefix}{key}:")
            for i, item in enumerate(value[:3], 1):  # Show first 3 items
                if isinstance(item, dict):
                    print(f"{prefix}  [{i}]")
                    print_result(item, indent + 2)
                else:
                    print(f"{prefix}  [{i}] {str(item)[:100]}")
            if len(value) > 3:
                print(f"{prefix}  ... ({len(value) - 3} more items)")
        else:
            print(f"{prefix}{key}: {value}")

async def test_mcp_connection_test():
    """Test MCP server connection"""
    print_section("TEST 1: MCP Server Connection")
    
    try:
        connected = await test_mcp_connection()
        if connected:
            print("✓ MCP Server is connected and working!")
            return True
        else:
            print("✗ MCP Server connection failed")
            return False
    except Exception as e:
        print(f"✗ Error connecting to MCP server: {e}")
        print("\nMake sure the MCP server is running:")
        print("  docker-compose up mcp-classifier")
        print("  OR")
        print("  python mcp_classifier_server.py")
        return False

async def test_broad_question():
    """Test a broad question that should use document summaries and map-reduce"""
    print_section("TEST 2: Broad Question (Document Summaries + Map-Reduce)")
    
    # Use global driver instance
    global _global_driver
    if _global_driver is None:
        _global_driver = GraphDatabase.driver(DB_URL, auth=(DB_USERNAME, DB_PASSWORD))
    
    # Create a fresh search pipeline instance for this test
    search_pipeline = UnifiedSearchPipeline(_global_driver)
    
    # Broad question that should trigger BROAD classification
    broad_question = "What are the main themes and topics discussed across all documents?"
    
    print(f"Question: {broad_question}\n")
    
    try:
        # First, test MCP classification
        print("Step 1: Testing MCP Classification...")
        classification = await classify_question_via_mcp(broad_question)
        print(f"  Classification Type: {classification['type']}")
        print(f"  Reason: {classification['reason']}")
        print(f"  Confidence: {classification['confidence']:.2f}\n")
        
        if classification['type'] != 'BROAD':
            print("⚠ Warning: Question was not classified as BROAD")
            print("  Expected: BROAD")
            print(f"  Got: {classification['type']}\n")
        
        # Now test the full search
        print("Step 2: Running Search (should use community summaries + map-reduce)...")
        result = await search_pipeline.search(
            question=broad_question,
            scope=SearchScope.HYBRID,
            max_chunks=7
        )
        
        print("\nSearch Results:")
        print(f"  Answer Length: {len(result.answer)} characters")
        print(f"  Confidence Score: {result.confidence_score:.2f}")
        print(f"  Question Type: {result.search_metadata.get('question_type', 'UNKNOWN')}")
        print(f"  Strategy: {result.search_metadata.get('strategy', 'UNKNOWN')}")
        print(f"  Communities Used: {result.search_metadata.get('communities_used', 0)}")
        print(f"  Chunks Used: {len(result.relevant_chunks)}")
        print(f"  Search Time: {result.search_metadata.get('search_time', 0):.2f}s")
        
        print("\nAnswer Preview:")
        print(f"  {result.answer[:500]}...")
        
        if result.community_summaries:
            print(f"\n  Community Summaries Retrieved: {len(result.community_summaries)}")
            for i, summary in enumerate(result.community_summaries[:3], 1):
                print(f"    [{i}] Community: {summary.get('community', 'N/A')}")
                print(f"        Similarity: {summary.get('similarity_score', 0):.2f}")
                print(f"        Summary: {summary.get('summary', '')[:150]}...")
        
        # Verify it used map-reduce
        if result.search_metadata.get('strategy') == 'map_reduce_communities':
            print("\n✓ SUCCESS: Map-reduce strategy was used!")
        else:
            print(f"\n⚠ Warning: Expected map_reduce_communities strategy, got: {result.search_metadata.get('strategy')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_specific_question():
    """Test a specific question that should use chunks and relations"""
    print_section("TEST 3: Specific Question (Chunks + Relations)")
    
    # Use global driver instance
    global _global_driver
    if _global_driver is None:
        _global_driver = GraphDatabase.driver(DB_URL, auth=(DB_USERNAME, DB_PASSWORD))
    
    # Create a fresh search pipeline instance for this test
    search_pipeline = UnifiedSearchPipeline(_global_driver)
    
    # Specific question that should trigger CHUNK classification
    specific_question = "What exact details are mentioned about the implementation process?"
    
    print(f"Question: {specific_question}\n")
    
    try:
        # First, test MCP classification
        print("Step 1: Testing MCP Classification...")
        classification = await classify_question_via_mcp(specific_question)
        print(f"  Classification Type: {classification['type']}")
        print(f"  Reason: {classification['reason']}")
        print(f"  Confidence: {classification['confidence']:.2f}\n")
        
        if classification['type'] != 'CHUNK':
            print("⚠ Warning: Question was not classified as CHUNK")
            print("  Expected: CHUNK")
            print(f"  Got: {classification['type']}\n")
        
        # Now test the full search
        print("Step 2: Running Search (should use chunk-level retrieval + relations)...")
        result = await search_pipeline.search(
            question=specific_question,
            scope=SearchScope.HYBRID,
            max_chunks=7
        )
        
        print("\nSearch Results:")
        print(f"  Answer Length: {len(result.answer)} characters")
        print(f"  Confidence Score: {result.confidence_score:.2f}")
        print(f"  Question Type: {result.search_metadata.get('question_type', 'UNKNOWN')}")
        print(f"  Strategy: {result.search_metadata.get('strategy', 'UNKNOWN')}")
        print(f"  Chunks Retrieved: {result.search_metadata.get('chunks_retrieved', 0)}")
        print(f"  Chunks Used: {len(result.relevant_chunks)}")
        print(f"  Entities Found: {len(result.entities_found)}")
        print(f"  Search Time: {result.search_metadata.get('search_time', 0):.2f}s")
        
        print("\nAnswer Preview:")
        print(f"  {result.answer[:500]}...")
        
        if result.relevant_chunks:
            print(f"\n  Chunks Retrieved: {len(result.relevant_chunks)}")
            for i, chunk in enumerate(result.relevant_chunks[:3], 1):
                chunk_id = chunk.get('chunk_id', 'N/A')
                chunk_id_str = str(chunk_id)[:20] if chunk_id != 'N/A' else 'N/A'
                print(f"    [{i}] Chunk ID: {chunk_id_str}...")
                print(f"        Similarity: {chunk.get('similarity_score', 0):.2f}")
                print(f"        Document: {chunk.get('metadata', {}).get('document_name', 'N/A')}")
                print(f"        Text Preview: {chunk.get('text', '')[:150]}...")
                if chunk.get('entities'):
                    print(f"        Entities: {', '.join(chunk.get('entities', [])[:3])}")
        
        if result.entities_found:
            print(f"\n  Entities Found: {len(result.entities_found)}")
            for i, entity in enumerate(result.entities_found[:5], 1):
                entity_name = entity.get('name', '') if isinstance(entity, dict) else str(entity)
                print(f"    [{i}] {entity_name}")
        
        # Verify it used chunk-level retrieval
        if result.search_metadata.get('strategy') == 'chunk_level_retrieval':
            print("\n✓ SUCCESS: Chunk-level retrieval strategy was used!")
        else:
            print(f"\n⚠ Warning: Expected chunk_level_retrieval strategy, got: {result.search_metadata.get('strategy')}")
        
        # Verify chunks were retrieved
        if len(result.relevant_chunks) > 0:
            print("✓ SUCCESS: Chunks were retrieved and used!")
        else:
            print("⚠ Warning: No chunks were retrieved")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    global _global_driver
    
    print("\n" + "=" * 80)
    print("  MCP Server Test Suite")
    print("=" * 80)
    
    results = []
    
    try:
        # Test 1: MCP Connection
        results.append(await test_mcp_connection_test())
        
        if not results[0]:
            print("\n⚠ MCP server is not available. Please start it first:")
            print("  docker-compose up mcp-classifier")
            print("  OR")
            print("  python mcp_classifier_server.py")
            return
        
        # Test 2: Broad Question
        results.append(await test_broad_question())
        
        # Test 3: Specific Question
        results.append(await test_specific_question())
        
    finally:
        # Cleanup: Close the global driver
        if _global_driver is not None:
            try:
                _global_driver.close()
                print("\n✓ Neo4j driver closed successfully")
            except Exception as e:
                print(f"\n⚠ Warning: Error closing Neo4j driver: {e}")
            finally:
                _global_driver = None
    
    # Summary
    print_section("Test Summary")
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed. Please review the output above.")

if __name__ == "__main__":
    asyncio.run(main())

