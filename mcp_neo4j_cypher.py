"""
MCP Neo4j Cypher Client for Graph/Analytical Queries

This module provides integration with the MCP Neo4j Cypher server to:
1. Retrieve graph schema (get-neo4j-schema)
2. Generate Cypher queries from natural language using LLM
3. Execute queries via MCP (read-neo4j-cypher)
4. Iteratively refine queries based on execution feedback

This module uses:
- config.py: Centralized configuration (MCP Neo4j settings, LLM settings)
- logging_config.py: Standardized structured logging
- resilience.py: Automatic retries and circuit breaking
- utils.py: LLM client for query generation
"""

import json
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# Import centralized configuration and logging
from config import get_config
from logging_config import get_logger, log_function_call, log_error_with_context
from utils import llm_client

# Get configuration
cfg = get_config()

# Setup logging
logger = get_logger("MCPNeo4jCypher")


@dataclass
class CypherQueryResult:
    """Result of a Cypher query execution"""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    query: Optional[str] = None


class MCPNeo4jCypherClient:
    """
    Client for interacting with MCP Neo4j Cypher server.
    
    Provides methods to:
    - Get graph schema
    - Generate Cypher queries from natural language
    - Execute queries with iterative refinement
    """
    
    def __init__(self, mcp_url: Optional[str] = None, mcp_timeout: Optional[int] = None, neo4j_driver=None):
        """
        Initialize MCP Neo4j Cypher client.
        
        Args:
            mcp_url: URL for MCP Neo4j Cypher server (defaults to config)
            mcp_timeout: Timeout for MCP requests (defaults to config)
            neo4j_driver: Optional Neo4j driver for direct queries (fallback)
        """
        self.logger = get_logger("MCPNeo4jCypherClient")
        self.neo4j_driver = neo4j_driver
        
        # Get MCP Neo4j configuration from config
        self.mcp_url = mcp_url or cfg.mcp_neo4j.url
        self.mcp_timeout = mcp_timeout or cfg.mcp_neo4j.timeout
        self.max_refinement_iterations = cfg.mcp_neo4j.max_refinement_iterations
        
        self.logger.info(
            f"MCP Neo4j Cypher client initialized",
            extra={"mcp_url": self.mcp_url, "timeout": self.mcp_timeout}
        )
    
    async def get_neo4j_schema(self) -> str:
        """
        Retrieve the Neo4j graph schema using MCP get-neo4j-schema tool.
        
        Returns:
            Schema description as a string
        """
        try:
            # Call MCP tool: get-neo4j-schema
            # This would typically be an HTTP call to the MCP server
            # For now, we'll implement a direct Neo4j schema query as fallback
            # In production, this should call the MCP server endpoint
            
            schema = await self._get_schema_via_mcp()
            if schema:
                return schema
            
            # Fallback: query schema directly from Neo4j
            return await self._get_schema_direct(self.neo4j_driver)
            
        except Exception as e:
            self.logger.error(f"Error retrieving Neo4j schema: {e}")
            return ""
    
    async def _get_schema_via_mcp(self) -> Optional[str]:
        """
        Get schema via MCP server (get-neo4j-schema tool).
        This is a placeholder - actual implementation depends on MCP server setup.
        """
        try:
            import httpx
            
            # MCP tool call format (adjust based on your MCP server implementation)
            async with httpx.AsyncClient(timeout=self.mcp_timeout) as client:
                response = await client.post(
                    f"{self.mcp_url}/tools/get-neo4j-schema",
                    json={"method": "get-neo4j-schema"}
                )
                if response.status_code == 200:
                    result = response.json()
                    return result.get("content", {}).get("text", "")
        except Exception as e:
            self.logger.warning(f"MCP schema retrieval failed: {e}, using direct query")
            return None
    
    async def _get_schema_direct(self, driver=None) -> str:
        """
        Get schema directly from Neo4j database.
        This queries the database for labels, relationships, and properties.
        """
        try:
            from utils import run_cypher_query_async
            
            if driver is None:
                # Try to get driver from document_api module
                try:
                    from document_api import driver
                except ImportError:
                    self.logger.error("Neo4j driver not available for direct schema query")
                    return "Neo4j Graph Schema: Unable to retrieve schema (driver not available)."
            
            # Query for node labels
            labels_query = """
            CALL db.labels() YIELD label
            RETURN collect(label) AS labels
            """
            
            # Query for relationship types
            rels_query = """
            CALL db.relationshipTypes() YIELD relationshipType
            RETURN collect(relationshipType) AS relationshipTypes
            """
            
            # Query for property keys
            props_query = """
            CALL db.propertyKeys() YIELD propertyKey
            RETURN collect(propertyKey) AS propertyKeys
            """
            
            labels_result = await run_cypher_query_async(driver, labels_query, {})
            rels_result = await run_cypher_query_async(driver, rels_query, {})
            props_result = await run_cypher_query_async(driver, props_query, {})
            
            labels = labels_result[0].get("labels", []) if labels_result else []
            rels = rels_result[0].get("relationshipTypes", []) if rels_result else []
            props = props_result[0].get("propertyKeys", []) if props_result else []
            
            # Build schema description
            schema_parts = [
                "Neo4j Graph Schema:",
                f"\nNode Labels: {', '.join(labels)}",
                f"\nRelationship Types: {', '.join(rels)}",
                f"\nProperty Keys: {', '.join(props[:20])}..." if len(props) > 20 else f"\nProperty Keys: {', '.join(props)}"
            ]
            
            # Get sample relationships between labels
            sample_query = """
            MATCH (a)-[r]->(b)
            WHERE labels(a) <> [] AND labels(b) <> []
            WITH DISTINCT labels(a)[0] AS fromLabel, type(r) AS relType, labels(b)[0] AS toLabel
            RETURN fromLabel, relType, toLabel
            LIMIT 20
            """
            sample_result = await run_cypher_query_async(driver, sample_query, {})
            
            if sample_result:
                schema_parts.append("\n\nSample Relationships:")
                for row in sample_result:
                    schema_parts.append(
                        f"  ({row['fromLabel']})-[:{row['relType']}]->({row['toLabel']})"
                    )
            
            return "\n".join(schema_parts)
            
        except Exception as e:
            self.logger.error(f"Error getting schema directly: {e}")
            return "Neo4j Graph Schema: Unable to retrieve schema details."
    
    async def generate_cypher_query(
        self,
        question: str,
        schema_context: str,
        refinement_feedback: Optional[str] = None
    ) -> str:
        """
        Generate a Cypher query from natural language using LLM.
        
        Args:
            question: User's question in natural language
            schema_context: Graph schema information
            refinement_feedback: Optional feedback from previous query execution
            
        Returns:
            Generated Cypher query string
        """
        try:
            if refinement_feedback:
                system_prompt = f"""You are a Cypher query expert. The last query you generated had an issue: {refinement_feedback}

Refine the Cypher query to correctly answer the question. Use only the schema elements provided below. Do not invent new labels or relationships.

Graph Schema:
{schema_context}

Instructions:
- Use only the labels, relationships, and properties from the schema above
- Return ONLY the Cypher query, nothing else
- Do not include markdown code blocks
- Ensure the query is syntactically correct
- If the question asks for a count, use COUNT()
- If the question asks for a list, return appropriate fields
- Use LIMIT to avoid returning too many results (default: 10)
"""
            else:
                system_prompt = f"""You are a Cypher query expert. Generate a Cypher query to answer the user's question using the provided graph schema.

Graph Schema:
{schema_context}

Instructions:
- Use only the labels, relationships, and properties from the schema above
- Do not invent new labels or relationships that don't exist
- Return ONLY the Cypher query, nothing else
- Do not include markdown code blocks or explanations
- Ensure the query is syntactically correct
- If the question asks for a count, use COUNT()
- If the question asks for a list, return appropriate fields
- Use LIMIT to avoid returning too many results (default: 10)
- Use at most 2-hop traversals unless absolutely necessary
"""
            
            user_prompt = f"Question: {question}\n\nCypher Query:"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await llm_client.invoke(messages)
            
            # Extract Cypher query from response (remove markdown if present)
            cypher_query = self._extract_cypher_query(response)
            
            self.logger.info(
                f"Generated Cypher query",
                extra={"question": question[:100], "query_length": len(cypher_query)}
            )
            
            return cypher_query
            
        except Exception as e:
            self.logger.error(f"Error generating Cypher query: {e}")
            raise
    
    def _extract_cypher_query(self, response: str) -> str:
        """Extract Cypher query from LLM response, removing markdown if present."""
        # Remove markdown code blocks
        response = re.sub(r'```cypher\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        response = response.strip()
        
        # If response contains multiple lines, take the first complete query
        lines = response.split('\n')
        query_lines = []
        in_query = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Check if line looks like Cypher (starts with MATCH, RETURN, CALL, etc.)
            if re.match(r'^(MATCH|RETURN|CALL|WITH|UNWIND|CREATE|MERGE|DELETE|SET|REMOVE)', line, re.IGNORECASE):
                in_query = True
                query_lines.append(line)
            elif in_query:
                query_lines.append(line)
                # Stop at empty line or non-Cypher line (unless it's a continuation)
                if line.endswith(';'):
                    break
        
        if query_lines:
            query = ' '.join(query_lines)
            # Remove trailing semicolon if present
            query = query.rstrip(';').strip()
            return query
        
        return response
    
    async def execute_cypher_query(self, cypher_query: str) -> CypherQueryResult:
        """
        Execute a Cypher query via MCP read-neo4j-cypher tool.
        
        Args:
            cypher_query: Cypher query string to execute
            
        Returns:
            CypherQueryResult with success status, data, or error message
        """
        try:
            # Call MCP tool: read-neo4j-cypher
            result = await self._execute_via_mcp(cypher_query)
            if result:
                return result
            
            # Fallback: execute directly via Neo4j driver
            return await self._execute_direct(cypher_query, self.neo4j_driver)
            
        except Exception as e:
            self.logger.error(f"Error executing Cypher query: {e}")
            return CypherQueryResult(
                success=False,
                error_message=str(e),
                query=cypher_query
            )
    
    async def _execute_via_mcp(self, cypher_query: str) -> Optional[CypherQueryResult]:
        """
        Execute query via MCP server (read-neo4j-cypher tool).
        This is a placeholder - actual implementation depends on MCP server setup.
        """
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=self.mcp_timeout) as client:
                response = await client.post(
                    f"{self.mcp_url}/tools/read-neo4j-cypher",
                    json={
                        "method": "read-neo4j-cypher",
                        "params": {"cypher": cypher_query}
                    }
                )
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("content", {})
                    
                    if "error" in content:
                        return CypherQueryResult(
                            success=False,
                            error_message=content["error"],
                            query=cypher_query
                        )
                    else:
                        data = content.get("data", [])
                        return CypherQueryResult(
                            success=True,
                            data=data,
                            query=cypher_query
                        )
        except Exception as e:
            self.logger.warning(f"MCP query execution failed: {e}, using direct execution")
            return None
    
    async def _execute_direct(self, cypher_query: str, driver=None) -> CypherQueryResult:
        """
        Execute query directly via Neo4j driver (fallback).
        """
        try:
            from utils import run_cypher_query_async
            
            if driver is None:
                # Try to get driver from document_api module
                try:
                    from document_api import driver
                except ImportError:
                    return CypherQueryResult(
                        success=False,
                        error_message="Neo4j driver not available",
                        query=cypher_query
                    )
            
            results = await run_cypher_query_async(driver, cypher_query, {})
            
            return CypherQueryResult(
                success=True,
                data=results,
                query=cypher_query
            )
            
        except Exception as e:
            error_msg = str(e)
            self.logger.warning(f"Direct query execution failed: {error_msg}")
            return CypherQueryResult(
                success=False,
                error_message=error_msg,
                query=cypher_query
            )
    
    def _is_result_relevant(self, result: CypherQueryResult, question: str) -> bool:
        """
        Check if query result is relevant to the question.
        
        Args:
            result: Query execution result
            question: Original question
            
        Returns:
            True if result seems relevant, False otherwise
        """
        if not result.success:
            return False
        
        if result.data is None:
            return False
        
        # Empty result might be valid (e.g., "no products out of stock")
        # But if question expects data, empty might indicate wrong query
        if len(result.data) == 0:
            # Check if question expects non-empty result
            q_lower = question.lower()
            if any(word in q_lower for word in ["list", "find", "show", "get", "which", "who"]):
                # Question expects results, empty might be wrong
                return False
        
        return True
    
    async def answer_with_cypher(
        self,
        question: str,
        max_iterations: Optional[int] = None
    ) -> Tuple[str, Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Answer a question using Cypher query generation and iterative refinement.
        
        This is the main entry point that:
        1. Gets graph schema
        2. Generates initial Cypher query
        3. Executes and refines iteratively
        4. Formats final answer
        
        Args:
            question: User's question
            max_iterations: Maximum refinement iterations (defaults to config)
            
        Returns:
            Tuple of (answer_text, query_results, metadata)
        """
        max_iters = max_iterations or self.max_refinement_iterations
        metadata = {
            "iterations": 0,
            "final_query": None,
            "refinement_history": []
        }
        
        try:
            # Step 1: Get graph schema
            self.logger.info("Retrieving Neo4j schema")
            schema_context = await self.get_neo4j_schema()
            
            if not schema_context:
                return (
                    "I'm sorry, I couldn't retrieve the graph schema. Please ensure Neo4j is accessible.",
                    None,
                    metadata
                )
            
            # Step 2: Generate initial Cypher query
            cypher_query = await self.generate_cypher_query(question, schema_context)
            metadata["final_query"] = cypher_query
            
            # Step 3: Iterative refinement loop
            for attempt in range(1, max_iters + 1):
                metadata["iterations"] = attempt
                
                self.logger.info(f"Executing Cypher query (attempt {attempt}/{max_iters})")
                
                # Execute query
                result = await self.execute_cypher_query(cypher_query)
                
                # Check if result is satisfactory
                if result.success and self._is_result_relevant(result, question):
                    # Success! Format the answer
                    answer = self._format_answer(question, result.data)
                    metadata["final_query"] = cypher_query
                    return answer, result.data, metadata
                
                # If we reach here, query needs refinement
                if attempt < max_iters:
                    feedback = result.error_message if result.error_message else "Query returned no relevant results"
                    metadata["refinement_history"].append({
                        "attempt": attempt,
                        "query": cypher_query,
                        "feedback": feedback
                    })
                    
                    self.logger.info(f"Refining query based on feedback: {feedback}")
                    
                    # Generate refined query
                    cypher_query = await self.generate_cypher_query(
                        question, schema_context, refinement_feedback=feedback
                    )
                else:
                    # Max iterations reached
                    if result.success:
                        # Query executed but might not be relevant
                        answer = self._format_answer(question, result.data)
                        return answer, result.data, metadata
                    else:
                        return (
                            f"I'm sorry, I couldn't generate a valid Cypher query to answer your question after {max_iters} attempts. "
                            f"Error: {result.error_message}",
                            None,
                            metadata
                        )
            
            # Should not reach here, but just in case
            return (
                "I'm sorry, I couldn't find an answer to your question in the graph data.",
                None,
                metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error in answer_with_cypher: {e}")
            return (
                f"I encountered an error while processing your graph query: {str(e)}",
                None,
                metadata
            )
    
    def _format_answer(
        self,
        question: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """
        Format Cypher query results into a natural language answer.
        
        Args:
            question: Original question
            results: Query results from Neo4j
            
        Returns:
            Formatted answer string
        """
        if not results:
            return "I couldn't find any results for your question in the graph data."
        
        # If result is a single count/number
        if len(results) == 1:
            result = results[0]
            keys = list(result.keys())
            if len(keys) == 1:
                value = result[keys[0]]
                if isinstance(value, (int, float)):
                    return f"The answer is {value}."
                elif isinstance(value, str) and len(value) < 100:
                    return f"The answer is {value}."
        
        # If result is a list of items
        if len(results) <= 10:
            # Format as a list
            answer_parts = ["Here are the results:"]
            for i, result in enumerate(results, 1):
                # Format result as key-value pairs
                formatted = ", ".join([f"{k}: {v}" for k, v in result.items()])
                answer_parts.append(f"{i}. {formatted}")
            return "\n".join(answer_parts)
        else:
            # Too many results, summarize
            return f"I found {len(results)} results. Here are the first few:\n" + "\n".join([
                f"{i}. {', '.join([f'{k}: {v}' for k, v in result.items()])}"
                for i, result in enumerate(results[:5], 1)
            ])


# Global instance
_mcp_client: Optional[MCPNeo4jCypherClient] = None


def get_mcp_neo4j_client(neo4j_driver=None) -> MCPNeo4jCypherClient:
    """Get or create the global MCP Neo4j Cypher client instance"""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPNeo4jCypherClient(neo4j_driver=neo4j_driver)
    return _mcp_client

