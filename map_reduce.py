"""
Map-Reduce Module for Community Summaries

Implements map-reduce pattern for processing large sets of community summaries
to generate comprehensive answers for broad questions.
"""

from typing import List, Dict, Any, Optional

# Import centralized configuration and logging
from config import get_config
from logging_config import get_logger, log_function_call
from utils import llm_client

# Get configuration
cfg = get_config()

# Setup logging using centralized logging config
logger = get_logger("MapReduce")

# Configuration from centralized config
MAP_MAX_COMMUNITIES = cfg.map_reduce.max_communities
MAP_BATCH_SIZE = cfg.map_reduce.batch_size
MAP_MIN_RELEVANCE = cfg.map_reduce.min_relevance

class MapReduceProcessor:
    """
    Map-Reduce processor for community summaries.
    
    Map Step: Extract relevant information from each community summary
    Reduce Step: Combine partial answers into a comprehensive final answer
    """
    
    def __init__(self):
        self.logger = get_logger("MapReduceProcessor")
    
    async def process_communities(
        self,
        question: str,
        communities: List[Dict[str, Any]],
        conversation_history: Optional[str] = None
    ) -> str:
        """
        Process communities using map-reduce pattern.
        
        Args:
            question: User's question
            communities: List of community summaries with structure:
                [{"summary": "...", "community": "...", "similarity_score": 0.8}, ...]
            conversation_history: Optional conversation context
            
        Returns:
            Final combined answer
        """
        if not communities:
            return "I couldn't find relevant information in the knowledge base to answer this question."
        
        # Limit communities to reasonable number
        communities = communities[:MAP_MAX_COMMUNITIES]
        
        # Filter by minimum relevance threshold
        filtered_communities = [
            c for c in communities
            if c.get("similarity_score", 0) >= MAP_MIN_RELEVANCE
        ]
        
        if not filtered_communities:
            return "I couldn't find sufficiently relevant information in the knowledge base."
        
        self.logger.info(f"Processing {len(filtered_communities)} communities with map-reduce")
        
        # Map step: Get partial answers from each community
        partial_answers = await self._map_step(question, filtered_communities, conversation_history)
        
        # Filter out empty or irrelevant partial answers
        valid_partials = [
            pa for pa in partial_answers
            if pa.get("answer") and pa["answer"].strip().upper() != "NO_RELEVANT_INFO"
        ]
        
        if not valid_partials:
            return "I couldn't find relevant information in the knowledge base to answer this question."
        
        # Reduce step: Combine partial answers
        final_answer = await self._reduce_step(question, valid_partials, conversation_history)
        
        return final_answer
    
    async def _map_step(
        self,
        question: str,
        communities: List[Dict[str, Any]],
        conversation_history: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Map step: Extract relevant information from each community.
        
        Returns:
            List of partial answers with metadata
        """
        partial_answers = []
        
        # Process in batches to avoid overwhelming the LLM
        for i in range(0, len(communities), MAP_BATCH_SIZE):
            batch = communities[i:i + MAP_BATCH_SIZE]
            
            # Process batch (can be parallelized in future)
            for community in batch:
                partial = await self._process_single_community(question, community, conversation_history)
                if partial:
                    partial_answers.append(partial)
        
        return partial_answers
    
    async def _process_single_community(
        self,
        question: str,
        community: Dict[str, Any],
        conversation_history: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single community to extract relevant information.
        """
        try:
            summary_text = community.get("summary", "")
            community_id = community.get("community", "unknown")
            similarity = community.get("similarity_score", 0.0)
            
            if not summary_text or not summary_text.strip():
                return None
            
            # Build context for map step
            context_parts = []
            if conversation_history:
                context_parts.append(f"Previous conversation:\n{conversation_history}\n")
            
            context_parts.append(f"Community Summary:\n{summary_text}")
            
            context = "\n".join(context_parts)
            
            # Map prompt: Extract relevant information
            map_prompt = f"""You are extracting relevant information from a community summary to help answer a question.

{context}

Question: {question}

Using ONLY the community summary above, extract any information that helps answer the question.
- Focus on information directly relevant to the question
- Be concise but complete
- If nothing in the summary is relevant, respond with exactly "NO_RELEVANT_INFO"
- Do not add information not present in the summary

Extracted relevant information:"""

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at extracting relevant information from summaries. Be precise and only use information from the provided context."
                },
                {
                    "role": "user",
                    "content": map_prompt
                }
            ]
            
            answer = await llm_client.invoke(messages)
            answer = answer.strip()
            
            # Skip if no relevant info
            if not answer or answer.upper() == "NO_RELEVANT_INFO":
                return None
            
            return {
                "answer": answer,
                "community_id": community_id,
                "similarity_score": similarity,
                "summary": summary_text[:200] + "..." if len(summary_text) > 200 else summary_text
            }
            
        except Exception as e:
            self.logger.warning(f"Error processing community {community.get('community', 'unknown')}: {e}")
            return None
    
    async def _reduce_step(
        self,
        question: str,
        partial_answers: List[Dict[str, Any]],
        conversation_history: Optional[str]
    ) -> str:
        """
        Reduce step: Combine partial answers into a comprehensive final answer.
        """
        if not partial_answers:
            return "I couldn't find relevant information in the knowledge base."
        
        if len(partial_answers) == 1:
            # Single partial answer, just clean it up
            return partial_answers[0]["answer"]
        
        try:
            # Build context for reduce step
            context_parts = []
            if conversation_history:
                context_parts.append(f"Previous conversation:\n{conversation_history}\n")
            
            context_parts.append("Question:")
            context_parts.append(question)
            context_parts.append("")
            context_parts.append("Partial answers extracted from different topics:")
            context_parts.append("")
            
            for i, partial in enumerate(partial_answers, 1):
                context_parts.append(f"{i}. {partial['answer']}")
            
            context = "\n".join(context_parts)
            
            # Reduce prompt: Combine partial answers
            reduce_prompt = f"""You are combining partial answers from different topics/communities into a single, comprehensive answer.

{context}

Combine these partial answers into a single, coherent, non-redundant answer to the question:
- Synthesize information from all relevant partial answers
- Remove redundancy and contradictions
- Organize the answer logically
- If there are conflicts between sources, acknowledge them
- Be comprehensive but concise
- Only use information from the partial answers provided

Final comprehensive answer:"""

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at synthesizing information from multiple sources into coherent answers. Be thorough and accurate."
                },
                {
                    "role": "user",
                    "content": reduce_prompt
                }
            ]
            
            final_answer = await llm_client.invoke(messages)
            return final_answer.strip()
            
        except Exception as e:
            self.logger.error(f"Error in reduce step: {e}")
            # Fallback: concatenate partial answers
            return "\n\n".join([pa["answer"] for pa in partial_answers])

# Global instance
map_reduce_processor = MapReduceProcessor()

async def map_reduce_communities(
    question: str,
    communities: List[Dict[str, Any]],
    conversation_history: Optional[str] = None
) -> str:
    """
    Process communities using map-reduce pattern.
    
    Args:
        question: User's question
        communities: List of community summaries
        conversation_history: Optional conversation context
        
    Returns:
        Final combined answer
    """
    return await map_reduce_processor.process_communities(
        question, communities, conversation_history
    )

