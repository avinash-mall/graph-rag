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

Graph Schema:
{schema_context}

The previous query failed. Try a DIFFERENT approach:
1. If searching for specific entities, use FUZZY matching: WHERE toLower(e.name) CONTAINS toLower('term')
2. If no results, try a BROADER query without specific entity names
3. If looking for relationships, try matching any entity first: MATCH (e:Entity)-[r]-(other:Entity) RETURN e.name, type(r), other.name LIMIT 20
4. Try simpler queries that explore the data rather than looking for exact matches

Alternative patterns to try:
- List all entities matching a term: MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower('partial_term') RETURN e.name, e.type LIMIT 10
- Browse relationships: MATCH (e1:Entity)-[r]->(e2:Entity) RETURN e1.name, type(r), e2.name LIMIT 20
- Count entities by type: MATCH (e:Entity) RETURN e.type, COUNT(*) AS count ORDER BY count DESC

CRITICAL: 
- Use toLower() and CONTAINS for fuzzy matching
- ONLY use labels: Entity, Chunk, CommunitySummary
- Return ONLY the Cypher query, no explanations
"""
            else:
                system_prompt = f"""You are a Cypher query expert. Generate a Cypher query to answer the user's question.

Graph Schema:
{schema_context}

QUERY PATTERNS (use these exactly):
1. Count entities: MATCH (e:Entity) RETURN COUNT(e) AS total
2. List entities by name: MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower('term') RETURN e.name, e.type LIMIT 20
3. Find all connections for an entity: MATCH (e:Entity)-[r]-(other:Entity) WHERE toLower(e.name) CONTAINS toLower('term') RETURN e.name, type(r), other.name LIMIT 20
4. Multiple search terms (use OR): MATCH (e:Entity) WHERE toLower(e.name) CONTAINS 'term1' OR toLower(e.name) CONTAINS 'term2' RETURN e.name LIMIT 20
5. Explore relationships: MATCH (e1:Entity)-[r]->(e2:Entity) RETURN e1.name, type(r), e2.name LIMIT 30

CRITICAL RULES:
1. Labels: ONLY use Entity, Chunk, CommunitySummary
2. Relationships: ONLY use MENTIONED_IN, CO_OCCURS, RELATES_TO
3. ALWAYS use toLower() and CONTAINS for name matching - NEVER use =
4. For relationship queries, match ONE entity first, then return its connections
5. NEVER use AND to require BOTH entities to match - use OR or match just ONE entity
6. Return ONLY the raw Cypher query, no markdown or explanations
7. Use LIMIT 20 or higher for relationship queries
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
    
    def _extract_terms_from_question(self, question: str) -> List[str]:
        """Extract key terms from the question for entity searching."""
        import re
        # Remove common words and extract meaningful terms
        stop_words = {
            'how', 'is', 'are', 'was', 'were', 'what', 'which', 'who', 'whom',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'all', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very', 'can',
            'will', 'just', 'should', 'now', 'related', 'about', 'its', 'it', 'this',
            'that', 'these', 'those', 'does', 'do', 'did', 'has', 'have', 'had'
        }
        
        # Extract words, keeping multi-word phrases in quotes
        words = re.findall(r'"([^"]+)"|(\w+)', question.lower())
        terms = []
        for phrase, word in words:
            term = phrase if phrase else word
            if term and term not in stop_words and len(term) > 2:
                terms.append(term)
        
        return terms
    
    async def find_similar_entities(self, terms: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find entities with names similar to the given terms using fuzzy matching.
        Returns a dict mapping each term to matching entities.
        """
        matches: Dict[str, List[Dict[str, Any]]] = {}
        
        try:
            from utils import run_cypher_query_async
            
            for term in terms[:5]:  # Limit to 5 terms
                # Try fuzzy match with CONTAINS and partial matching
                query = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($term)
                   OR toLower($term) CONTAINS toLower(e.name)
                RETURN e.name AS name, e.type AS type, 
                       CASE WHEN toLower(e.name) = toLower($term) THEN 1.0
                            WHEN toLower(e.name) CONTAINS toLower($term) THEN 0.8
                            ELSE 0.6 END AS similarity
                ORDER BY similarity DESC
                LIMIT 5
                """
                
                results = await self._execute_direct(query.replace("$term", f"'{term}'"), self.neo4j_driver)
                if results.success and results.data:
                    matches[term] = [{"name": r["name"], "type": r.get("type", ""), "similarity": r.get("similarity", 0.5)} for r in results.data]
                else:
                    matches[term] = []
                    
        except Exception as e:
            self.logger.warning(f"Fuzzy entity search failed: {e}")
            
        return matches
    
    async def generate_multiple_cypher_queries(
        self, 
        question: str, 
        schema_context: str,
        terms: List[str],
        fuzzy_matches: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Generate 5 different Cypher query strategies to explore the question.
        
        Returns list of dicts with: type, query, description
        """
        queries = []
        
        # Get actual entity names from fuzzy matches
        matched_entities = []
        for term, matches in fuzzy_matches.items():
            if matches:
                matched_entities.append(matches[0]["name"])  # Best match
        
        # If no matches found, use original terms
        search_terms = matched_entities if matched_entities else terms
        
        # Strategy 1: Fuzzy search on all extracted terms
        if terms:
            term_conditions = " OR ".join([f"toLower(e.name) CONTAINS toLower('{t}')" for t in terms[:3]])
            queries.append({
                "type": "fuzzy_search",
                "description": f"Searching entities matching: {', '.join(terms[:3])}",
                "query": f"""
                    MATCH (e:Entity)
                    WHERE {term_conditions}
                    RETURN e.name AS entity, e.type AS type
                    LIMIT 10
                """
            })
        
        # Strategy 2: Relationship exploration between matched entities
        if len(search_terms) >= 2:
            e1, e2 = search_terms[0], search_terms[1]
            queries.append({
                "type": "relationship_exploration",
                "description": f"Finding relationships between '{e1}' and '{e2}'",
                "query": f"""
                    MATCH (e1:Entity)-[r]-(e2:Entity)
                    WHERE (toLower(e1.name) CONTAINS toLower('{e1}') OR toLower(e1.name) = toLower('{e1}'))
                      AND (toLower(e2.name) CONTAINS toLower('{e2}') OR toLower(e2.name) = toLower('{e2}'))
                    RETURN e1.name AS from_entity, type(r) AS relationship, e2.name AS to_entity
                    LIMIT 15
                """
            })
        
        # Strategy 3: Path finding via intermediate nodes
        if len(search_terms) >= 2:
            e1, e2 = search_terms[0], search_terms[1]
            queries.append({
                "type": "path_finding",
                "description": f"Finding paths connecting '{e1}' to '{e2}'",
                "query": f"""
                    MATCH (e1:Entity), (e2:Entity)
                    WHERE toLower(e1.name) CONTAINS toLower('{e1}')
                      AND toLower(e2.name) CONTAINS toLower('{e2}')
                    MATCH path = (e1)-[*1..3]-(e2)
                    RETURN [n IN nodes(path) | CASE WHEN 'Entity' IN labels(n) THEN n.name ELSE 'chunk' END] AS path_nodes,
                           length(path) AS path_length
                    LIMIT 10
                """
            })
        
        # Strategy 4: Related entities via RELATES_TO for first term
        if search_terms:
            e1 = search_terms[0]
            queries.append({
                "type": "related_entities",
                "description": f"Finding entities related to '{e1}'",
                "query": f"""
                    MATCH (e1:Entity)-[:RELATES_TO]-(related:Entity)
                    WHERE toLower(e1.name) CONTAINS toLower('{e1}')
                    RETURN e1.name AS source_entity, related.name AS related_entity, related.type AS related_type
                    LIMIT 20
                """
            })
        
        # Strategy 5: Chunks containing the entities (for context)
        if search_terms:
            term_conditions = " OR ".join([f"toLower(e.name) CONTAINS toLower('{t}')" for t in search_terms[:2]])
            queries.append({
                "type": "chunk_context",
                "description": f"Finding document chunks mentioning these entities",
                "query": f"""
                    MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
                    WHERE {term_conditions}
                    RETURN e.name AS entity, c.text AS chunk_text, c.document_name AS document
                    LIMIT 5
                """
            })
        
        return queries
    
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
        Answer a question using multiple Cypher query strategies.
        
        This enhanced version:
        1. Extracts key terms from the question
        2. Finds fuzzy entity matches for those terms
        3. Generates 5 different query strategies
        4. Executes all queries and combines results
        5. Formats a comprehensive answer
        
        Args:
            question: User's question
            max_iterations: Maximum refinement iterations (defaults to config)
            
        Returns:
            Tuple of (answer_text, query_results, metadata)
        """
        import asyncio
        
        metadata = {
            "iterations": 0,
            "final_query": None,
            "query_strategies": [],
            "fuzzy_matches": {},
            "extracted_terms": []
        }
        
        try:
            # Step 1: Extract key terms from the question
            self.logger.info("Extracting terms from question")
            terms = self._extract_terms_from_question(question)
            metadata["extracted_terms"] = terms
            self.logger.info(f"Extracted terms: {terms}")
            
            if not terms:
                return (
                    "I couldn't identify specific topics in your question. Could you please rephrase with more specific terms?",
                    None,
                    metadata
                )
            
            # Step 2: Find fuzzy entity matches
            self.logger.info("Finding similar entities via fuzzy matching")
            fuzzy_matches = await self.find_similar_entities(terms)
            metadata["fuzzy_matches"] = {k: [m["name"] for m in v] for k, v in fuzzy_matches.items()}
            
            # Log what we found
            for term, matches in fuzzy_matches.items():
                if matches:
                    self.logger.info(f"Term '{term}' matched entities: {[m['name'] for m in matches]}")
            
            # Step 3: Get schema context for LLM fallback
            schema_context = await self.get_neo4j_schema()
            
            # Step 4: Generate multiple query strategies
            self.logger.info("Generating multiple Cypher query strategies")
            query_strategies = await self.generate_multiple_cypher_queries(
                question, schema_context, terms, fuzzy_matches
            )
            
            # Step 5: Execute all queries and collect results
            all_results: List[Dict[str, Any]] = []
            query_outcomes = []
            
            for strategy in query_strategies:
                self.logger.info(f"Executing strategy: {strategy['type']}")
                result = await self.execute_cypher_query(strategy["query"])
                
                outcome = {
                    "type": strategy["type"],
                    "description": strategy["description"],
                    "success": result.success,
                    "result_count": len(result.data) if result.data else 0,
                    "query": strategy["query"].strip()
                }
                query_outcomes.append(outcome)
                
                if result.success and result.data:
                    # Tag results with their source strategy
                    for r in result.data:
                        r["_strategy"] = strategy["type"]
                    all_results.extend(result.data)
                    self.logger.info(f"Strategy '{strategy['type']}' returned {len(result.data)} results")
            
            metadata["query_strategies"] = query_outcomes
            metadata["iterations"] = len(query_strategies)
            
            # Find the most successful query for metadata
            successful_queries = [q for q in query_outcomes if q["result_count"] > 0]
            if successful_queries:
                best_query = max(successful_queries, key=lambda x: x["result_count"])
                metadata["final_query"] = best_query["query"]
            
            # Step 6: If no results from structured queries, try LLM-generated query
            if not all_results:
                self.logger.info("No results from structured queries, trying LLM-generated query")
                cypher_query = await self.generate_cypher_query(question, schema_context)
                result = await self.execute_cypher_query(cypher_query)
                
                if result.success and result.data:
                    all_results.extend(result.data)
                    metadata["final_query"] = cypher_query
                    query_outcomes.append({
                        "type": "llm_generated",
                        "description": "LLM-generated fallback query",
                        "success": True,
                        "result_count": len(result.data),
                        "query": cypher_query
                    })
            
            # Step 7: Format the comprehensive answer
            if all_results:
                # Deduplicate results
                unique_results = self._deduplicate_results(all_results)
                answer = await self._format_comprehensive_answer(
                    question, unique_results, terms, fuzzy_matches, query_outcomes
                )
                return answer, unique_results, metadata
            else:
                # No results found
                answer = await self._format_no_results_answer(question, terms, fuzzy_matches)
                return answer, None, metadata
            
        except Exception as e:
            self.logger.error(f"Error in answer_with_cypher: {e}")
            return (
                f"I encountered an error while processing your graph query: {str(e)}",
                None,
                metadata
            )
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on key fields."""
        seen = set()
        unique = []
        for r in results:
            # Create a hashable key from non-strategy fields
            key_fields = {k: v for k, v in r.items() if k != "_strategy" and isinstance(v, (str, int, float, bool))}
            key = tuple(sorted(key_fields.items()))
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique
    
    async def _format_comprehensive_answer(
        self,
        question: str,
        results: List[Dict[str, Any]],
        terms: List[str],
        fuzzy_matches: Dict[str, List[Dict[str, Any]]],
        query_outcomes: List[Dict[str, Any]]
    ) -> str:
        """Format a comprehensive, natural language answer from multi-query results."""
        
        # Build context about what was found
        fuzzy_context = []
        for term, matches in fuzzy_matches.items():
            if matches:
                fuzzy_context.append(f"'{term}' matched entity '{matches[0]['name']}'")
        
        successful_strategies = [q["type"] for q in query_outcomes if q["result_count"] > 0]
        
        # Format results for LLM
        results_text = json.dumps(results[:20], indent=2, default=str)
        
        prompt = f"""Answer this question based on the graph database results. Be conversational and helpful.

QUESTION: {question}

FUZZY MATCHING:
{chr(10).join(fuzzy_context) if fuzzy_context else "No fuzzy matches needed - entities found directly."}

SUCCESSFUL QUERY STRATEGIES: {', '.join(successful_strategies) if successful_strategies else 'None'}

GRAPH QUERY RESULTS:
{results_text}

INSTRUCTIONS:
1. Write a natural, conversational response (3-5 sentences)
2. Directly address the question the user asked
3. Mention specific entities and relationships found in the results
4. If the user's spelling was corrected (fuzzy match), acknowledge this naturally
5. Explain HOW the entities are connected based on the relationships found
6. If no direct connection was found, explain what WAS found and what it might mean
7. Do NOT just list raw data - synthesize it into a meaningful answer

ANSWER:"""

        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that explains graph database findings in clear, natural language. Be conversational and insightful."},
                {"role": "user", "content": prompt}
            ]
            
            answer = await llm_client.invoke(messages)
            
            if answer and len(answer.strip()) > 30:
                return answer.strip()
            else:
                return self._simple_format(question, results)
                
        except Exception as e:
            self.logger.warning(f"LLM formatting failed: {e}")
            return self._simple_format(question, results)
    
    async def _format_no_results_answer(
        self,
        question: str,
        terms: List[str],
        fuzzy_matches: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Generate a helpful response when no results were found."""
        
        unmatched_terms = [t for t in terms if not fuzzy_matches.get(t)]
        matched_terms = {t: fuzzy_matches[t][0]["name"] for t in terms if fuzzy_matches.get(t)}
        
        if unmatched_terms:
            return (
                f"I couldn't find entities matching '{', '.join(unmatched_terms)}' in the knowledge graph. "
                f"This might mean the entity isn't in the database or has a different name. "
                f"Try searching for related topics or check the spelling."
            )
        elif matched_terms:
            return (
                f"I found entities matching your terms ({', '.join(matched_terms.values())}), "
                f"but couldn't find any direct relationships between them in the graph. "
                f"They may not be directly connected in the current knowledge base."
            )
        else:
            return (
                "I couldn't find relevant information in the knowledge graph. "
                "Please try rephrasing your question or using different terms."
            )
    
    async def _format_answer(
        self,
        question: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """
        Format Cypher query results into a natural language answer using LLM.
        
        Args:
            question: Original question
            results: Query results from Neo4j
            
        Returns:
            Formatted natural language answer string
        """
        if not results:
            return "I couldn't find any results for your question in the graph data."
        
        # If result is a single count/number, return directly
        if len(results) == 1:
            result = results[0]
            keys = list(result.keys())
            if len(keys) == 1:
                value = result[keys[0]]
                if isinstance(value, (int, float)):
                    return f"The answer is {value}."
        
        # Use LLM to format results into natural language
        try:
            # Format results as JSON for the LLM
            results_text = json.dumps(results[:15], indent=2, default=str)  # Limit to 15 results
            
            prompt = f"""Based on the following graph database query results, provide a clear and helpful natural language answer to the user's question.

Question: {question}

Query Results (from Neo4j graph database):
{results_text}

Instructions:
1. Summarize the key findings in 2-4 sentences
2. Highlight the most relevant relationships or entities found
3. Be specific and cite actual entity names from the results
4. If the results show relationships, explain how the entities are connected
5. Do NOT just list the raw data - synthesize it into a meaningful answer

Answer:"""

            messages = [
                {"role": "system", "content": "You are a helpful assistant that explains graph database query results in clear, natural language. Be concise but informative."},
                {"role": "user", "content": prompt}
            ]
            
            answer = await llm_client.invoke(messages)
            
            if answer and len(answer.strip()) > 20:
                return answer.strip()
            else:
                # Fallback to simple formatting
                return self._simple_format(question, results)
                
        except Exception as e:
            self.logger.warning(f"LLM formatting failed: {e}, using simple format")
            return self._simple_format(question, results)
    
    def _simple_format(self, question: str, results: List[Dict[str, Any]]) -> str:
        """Simple fallback formatting without LLM."""
        if len(results) <= 10:
            answer_parts = ["Here are the results:"]
            for i, result in enumerate(results, 1):
                formatted = ", ".join([f"{k}: {v}" for k, v in result.items()])
                answer_parts.append(f"{i}. {formatted}")
            return "\n".join(answer_parts)
        else:
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

