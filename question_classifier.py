"""
Question Classifier - Agentic LLM-based question classification for routing.

Classifies questions into:
- BROAD: Needs broad understanding/overview → use community summaries
- CHUNK: Needs specific details → use chunk-level retrieval  
- OUT_OF_SCOPE: Not covered by knowledge base
"""

import logging
import os
import json
import re
from typing import Literal, TypedDict, Optional
from enum import Enum
from dotenv import load_dotenv

from utils import llm_client

load_dotenv()

logger = logging.getLogger("QuestionClassifier")

# MCP Configuration
USE_MCP_CLASSIFIER = os.getenv("USE_MCP_CLASSIFIER", "false").lower() == "true"

QuestionType = Literal["BROAD", "CHUNK", "OUT_OF_SCOPE"]

class ClassificationResult(TypedDict):
    """Result of question classification"""
    type: QuestionType
    reason: str
    confidence: float

class QuestionClassification(Enum):
    """Question classification types"""
    BROAD = "BROAD"
    CHUNK = "CHUNK"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"

class QuestionClassifier:
    """
    Agentic LLM-based question classifier for routing search strategies.
    
    Classifies questions to determine the best retrieval strategy:
    - BROAD: Use community summaries with map-reduce
    - CHUNK: Use chunk-level similarity search
    - OUT_OF_SCOPE: Polite fallback
    """
    
    def __init__(self):
        self.logger = logging.getLogger("QuestionClassifier")
        self.use_mcp = USE_MCP_CLASSIFIER
        self.use_heuristics = os.getenv("CLASSIFIER_USE_HEURISTICS", "true").lower() == "true"
        self.use_llm = os.getenv("CLASSIFIER_USE_LLM", "true").lower() == "true"
        
        if self.use_mcp:
            self.logger.info("MCP classifier enabled - will use MCP server for classification")
        
    async def classify(self, question: str) -> ClassificationResult:
        """
        Classify a question into one of the question types.
        
        Args:
            question: User's question
            
        Returns:
            ClassificationResult with type, reason, and confidence
        """
        if not question or not question.strip():
            return {
                "type": "OUT_OF_SCOPE",
                "reason": "Empty or invalid question",
                "confidence": 1.0
            }
        
        # Step 0: Try MCP classification if enabled
        if self.use_mcp:
            try:
                from mcp_classifier_client import classify_question_via_mcp
                mcp_result = await classify_question_via_mcp(question)
                if mcp_result and mcp_result.get("type"):
                    self.logger.info(f"MCP classification: {mcp_result['type']} - {mcp_result.get('reason', '')}")
                    return mcp_result
            except ImportError:
                self.logger.warning("MCP classifier client not available, falling back to direct classification")
            except Exception as e:
                self.logger.warning(f"MCP classification failed: {e}, falling back to direct classification")
        
        # Step 1: Try heuristics first (fast, cheap)
        if self.use_heuristics:
            heuristic_result = self._classify_with_heuristics(question)
            if heuristic_result["confidence"] >= 0.8:
                self.logger.info(f"High-confidence heuristic classification: {heuristic_result['type']}")
                return heuristic_result
        
        # Step 2: Use LLM for classification (more accurate)
        if self.use_llm:
            try:
                llm_result = await self._classify_with_llm(question)
                if llm_result:
                    self.logger.info(f"LLM classification: {llm_result['type']} - {llm_result['reason']}")
                    return llm_result
            except Exception as e:
                self.logger.warning(f"LLM classification failed: {e}, falling back to heuristics")
        
        # Step 3: Fallback to default heuristic result
        return self._classify_with_heuristics(question)
    
    def _classify_with_heuristics(self, question: str) -> ClassificationResult:
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
    
    async def _classify_with_llm(self, question: str) -> Optional[ClassificationResult]:
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
            result = self._parse_classification_response(response)
            
            if result and result.get("type") in ["BROAD", "CHUNK", "OUT_OF_SCOPE"]:
                return result
            else:
                self.logger.warning(f"Invalid LLM classification result: {result}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in LLM classification: {e}")
            return None
    
    def _parse_classification_response(self, response: str) -> Optional[ClassificationResult]:
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
            self.logger.warning(f"Failed to parse classification response: {e}")
            return None

# Global instance
classifier = QuestionClassifier()

async def classify_question(question: str) -> ClassificationResult:
    """
    Classify a question using the global classifier instance.
    
    Args:
        question: User's question
        
    Returns:
        ClassificationResult with type, reason, and confidence
    """
    return await classifier.classify(question)

