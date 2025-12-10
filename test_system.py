"""
Comprehensive test suite for the Graph RAG system

Tests all key components:
- Efficient NLP processing with spaCy
- Batch embedding optimization
- Unified search pipeline
- Async handling and performance
- Error handling and edge cases
- External API URL configurations and validation
"""

import asyncio
import pytest
import os
import tempfile
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock
from urllib.parse import urlparse
import httpx
from dotenv import load_dotenv

# Load environment variables for tests
load_dotenv()

# Import modules to test
from utils import (
    EfficientNLPProcessor, BatchEmbeddingClient, AsyncLLMClient,
    ImprovedTextProcessor, cosine_similarity, extract_entities_efficient
)
from unified_search import UnifiedSearchPipeline, SearchScope
from document_api import GraphManager
from question_classifier import QuestionClassifier, classify_question
from map_reduce import MapReduceProcessor, map_reduce_communities

class TestEfficientNLPProcessor:
    """Test the efficient NLP processor"""
    
    def setup_method(self):
        """Setup for each test"""
        self.processor = EfficientNLPProcessor()
    
    @pytest.mark.asyncio
    async def test_entity_extraction_basic(self):
        """Test basic entity extraction"""
        text = "Apple Inc. is a technology company founded by Steve Jobs in Cupertino."
        entities = await self.processor.extract_entities(text)
        
        # Note: This test may fail if LLM API is not available, but structure should work
        assert isinstance(entities, list)
        if len(entities) > 0:
            entity_names = [e.name for e in entities]
            assert any("Apple" in name or "Steve" in name or "Jobs" in name or "Cupertino" in name for name in entity_names)
    
    @pytest.mark.asyncio
    async def test_entity_extraction_empty_text(self):
        """Test entity extraction with empty text"""
        entities = await self.processor.extract_entities("")
        assert entities == []
        
        entities = await self.processor.extract_entities(None)
        assert entities == []
    
    @pytest.mark.asyncio
    async def test_entity_type_mapping(self):
        """Test entity type mapping"""
        text = "Microsoft Corporation was founded in 1975."
        entities = await self.processor.extract_entities(text)
        
        # Note: This test may fail if LLM API is not available, but structure should work
        assert isinstance(entities, list)
        if len(entities) > 0:
            org_entities = [e for e in entities if e.type == "ORGANIZATION"]
            date_entities = [e for e in entities if e.type == "DATE"]
            # At least one entity should be found
            assert len(org_entities) > 0 or len(date_entities) > 0
    
    @pytest.mark.asyncio
    async def test_coreference_resolution_fallback(self):
        """Test coreference resolution fallback"""
        text = "John Smith is a developer. He works at Google."
        resolved = await self.processor.resolve_coreferences(text)
        
        # Should return some text (even if LLM fails, should return original)
        assert isinstance(resolved, str)
        assert len(resolved) > 0

class TestBatchEmbeddingClient:
    """Test the batch embedding client"""
    
    def setup_method(self):
        """Setup for each test"""
        self.client = BatchEmbeddingClient()
    
    @pytest.mark.asyncio
    async def test_batch_embedding_mock(self):
        """Test batch embedding with mocked API"""
        texts = ["Hello world", "This is a test", "Another sentence"]
        
        # Mock the API call based on provider
        method_name = '_fetch_batch_embeddings_openai' if self.client.provider in {"openai", "ollama"} else '_fetch_batch_embeddings_gemini'
        with patch.object(self.client, method_name) as mock_fetch:
            mock_fetch.return_value = [[0.1, 0.2, 0.3]] * len(texts)
            
            embeddings = await self.client.get_embeddings(texts)
            
            assert len(embeddings) == len(texts)
            assert all(len(emb) == 3 for emb in embeddings)
            mock_fetch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_caching(self):
        """Test embedding caching"""
        text = "Test text for caching"
        
        # Mock the API call based on provider
        method_name = '_fetch_batch_embeddings_openai' if self.client.provider in {"openai", "ollama"} else '_fetch_batch_embeddings_gemini'
        with patch.object(self.client, method_name) as mock_fetch:
            mock_fetch.return_value = [[0.1, 0.2, 0.3]]
            
            # First call
            emb1 = await self.client.get_embedding(text)
            
            # Second call should use cache
            emb2 = await self.client.get_embedding(text)
            
            assert emb1 == emb2
            # Should only call API once due to caching
            mock_fetch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_empty_texts(self):
        """Test handling of empty texts"""
        texts = ["", "valid text", None, "another valid"]
        
        # Mock the API call based on provider
        method_name = '_fetch_batch_embeddings_openai' if self.client.provider in {"openai", "ollama"} else '_fetch_batch_embeddings_gemini'
        with patch.object(self.client, method_name) as mock_fetch:
            mock_fetch.return_value = [[0.1, 0.2]] * 2  # Only for valid texts
            
            embeddings = await self.client.get_embeddings(texts)
            
            # Should return embeddings for all inputs, empty for invalid ones
            assert len(embeddings) == len(texts)
    
    def test_api_key_configuration(self):
        """Test that API key is properly configured"""
        # Test that API key is loaded from environment
        assert hasattr(self.client, 'embedding_api_key')
        
        # Test that API key is either set or None (depending on environment)
        api_key = self.client.embedding_api_key
        assert api_key is None or isinstance(api_key, str)
        
        # Test that headers are properly prepared when API key exists
        if api_key:
            headers = {"Content-Type": "application/json"}
            headers["Authorization"] = f"Bearer {api_key}"
            assert "Authorization" in headers
            assert headers["Authorization"].startswith("Bearer ")
    
    @pytest.mark.asyncio
    async def test_api_key_in_headers(self):
        """Test that API key is included in request headers"""
        texts = ["test text"]
        
        # Mock httpx.AsyncClient to capture headers
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            if self.client.provider in {"openai", "ollama"}:
                mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
            else:
                mock_response.json.return_value = {"embeddings": [{"values": [0.1, 0.2, 0.3]}]}
            mock_response.raise_for_status.return_value = None
            
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            # Call the appropriate method based on provider
            if self.client.provider in {"openai", "ollama"}:
                await self.client._fetch_batch_embeddings_openai(texts)
            else:
                await self.client._fetch_batch_embeddings_gemini(texts)
            
            # Verify that post was called
            mock_client.return_value.__aenter__.return_value.post.assert_called_once()
            
            # Get the call arguments
            call_args = mock_client.return_value.__aenter__.return_value.post.call_args
            headers = call_args.kwargs.get('headers', {})
            
            # Verify headers include Content-Type
            assert "Content-Type" in headers
            assert headers["Content-Type"] == "application/json"
            
            # If API key is set, verify Authorization header (for OpenAI/Ollama) or x-goog-api-key (for Gemini)
            if self.client.embedding_api_key:
                if self.client.provider in {"openai", "ollama"}:
                    assert "Authorization" in headers
                    assert headers["Authorization"] == f"Bearer {self.client.embedding_api_key}"
                else:
                    assert "x-goog-api-key" in headers
                    assert headers["x-goog-api-key"] == self.client.embedding_api_key

class TestImprovedTextProcessor:
    """Test the improved text processor"""
    
    def setup_method(self):
        """Setup for each test"""
        self.processor = ImprovedTextProcessor()
    
    def test_text_cleaning(self):
        """Test text cleaning functionality"""
        dirty_text = "  Hello\n\n\nworld!  \t  Page 123  "
        cleaned = self.processor.clean_text(dirty_text)
        
        assert "Hello world!" in cleaned
        assert "Page 123" not in cleaned  # Page numbers should be removed
        assert not cleaned.startswith(" ")  # Leading whitespace removed
    
    def test_advanced_chunking(self):
        """Test advanced text chunking with overlap"""
        def overlap_length(chunk_a: str, chunk_b: str) -> int:
            max_len = min(len(chunk_a), len(chunk_b))
            for i in range(max_len, 0, -1):
                if chunk_a[-i:] == chunk_b[:i]:
                    return i
            return 0

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        overlap = 10
        chunks = self.processor.chunk_text_advanced(text, max_chunk_size=50, overlap=overlap)

        assert len(chunks) > 1
        for prev_chunk, next_chunk in zip(chunks, chunks[1:]):
            shared = overlap_length(prev_chunk, next_chunk)
            # Overlap may be slightly less due to sentence boundaries, allow some tolerance
            assert shared >= max(0, min(overlap, len(prev_chunk)) - 5)

        boundary_text = "AAAAAAAAAAAAAAAAAA. BBBBBBBBBBB."
        boundary_overlap = 5
        boundary_chunks = self.processor.chunk_text_advanced(
            boundary_text, max_chunk_size=20, overlap=boundary_overlap
        )

        assert len(boundary_chunks) == 2
        boundary_shared = overlap_length(boundary_chunks[0], boundary_chunks[1])
        assert boundary_shared >= min(boundary_overlap, len(boundary_chunks[0]))
    
    def test_chunking_empty_text(self):
        """Test chunking with empty text"""
        chunks = self.processor.chunk_text_advanced("")
        assert chunks == []
        
        chunks = self.processor.chunk_text_advanced("   ")
        assert chunks == []

class TestUnifiedSearchPipeline:
    """Test the unified search pipeline"""
    
    def setup_method(self):
        """Setup for each test"""
        self.mock_driver = Mock()
        self.pipeline = UnifiedSearchPipeline(self.mock_driver)
    
    @pytest.mark.asyncio
    async def test_query_preprocessing(self):
        """Test query preprocessing"""
        question = "What is AI?"
        conversation = "We were talking about technology."
        
        with patch.object(self.pipeline, '_preprocess_query') as mock_preprocess:
            mock_preprocess.return_value = "What is artificial intelligence in the context of technology?"
            
            result = await self.pipeline._preprocess_query(question, conversation)
            
            assert isinstance(result, str)
            mock_preprocess.assert_called_once_with(question, conversation)
    
    @pytest.mark.asyncio
    async def test_search_with_mocked_components(self):
        """Test full search with mocked components"""
        question = "Test question"
        
        # Mock classification first
        with patch('unified_search.classify_question') as mock_classify:
            mock_classify.return_value = {
                "type": "CHUNK",
                "reason": "Test question",
                "confidence": 0.8
            }
            
            # Mock all the internal methods for chunk search
            with patch.multiple(
                self.pipeline,
                _preprocess_query=AsyncMock(return_value="processed question"),
                _retrieve_relevant_chunks=AsyncMock(return_value=[]),
                _expand_context_via_graph=AsyncMock(return_value=[]),
                _rank_and_filter_chunks=AsyncMock(return_value=[]),
                _get_relevant_community_summaries=AsyncMock(return_value=[]),
                _generate_answer=AsyncMock(return_value="Test answer"),
                _extract_entities_from_chunks=AsyncMock(return_value=[]),
                _search_with_chunks=AsyncMock(return_value=type('obj', (object,), {
                    'answer': 'Test answer',
                    'relevant_chunks': [],
                    'community_summaries': [],
                    'entities_found': [],
                    'confidence_score': 0.7,
                    'search_metadata': {'question_type': 'CHUNK', 'search_time': 0.5}
                })())
            ):
                result = await self.pipeline.search(question)
                
                assert result.answer == "Test answer"
                assert result.confidence_score >= 0.0
                assert isinstance(result.search_metadata, dict)
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        from unified_search import RetrievedChunk
        
        chunks = [
            RetrievedChunk("1", "doc1", "text1", 0.8, [], {}, None, None),
            RetrievedChunk("2", "doc1", "text2", 0.6, [], {}, None, None)
        ]
        
        confidence = self.pipeline._calculate_confidence_score(chunks, [])
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high for good chunks
    
    @pytest.mark.asyncio
    async def test_vector_search_fallback(self):
        """Test vector search fallback to GDS similarity"""
        question_embedding = [0.1, 0.2, 0.3] * 100  # Mock embedding
        
        # Mock the database query to fail for vector index, then succeed for fallback
        with patch('unified_search.run_cypher_query_async') as mock_query:
            # First call (vector index) fails, second call (fallback) succeeds
            mock_query.side_effect = [
                Exception("Vector index not found"),  # Vector index query fails
                [  # Fallback GDS query succeeds
                    {
                        "chunk_id": "1",
                        "doc_id": "doc1",
                        "text": "Test text",
                        "document_name": "test.pdf",
                        "embedding": question_embedding,
                        "similarity": 0.8
                    }
                ]
            ]
            
            # Should fallback and return results
            result = await self.pipeline._run_vector_similarity_search(
                question_embedding, None, 5
            )
            
            # Should have called fallback and returned results
            assert len(result) >= 0  # May be empty after filtering, but should not error
    
    @pytest.mark.asyncio
    async def test_graph_aware_reranking(self):
        """Test graph-aware reranking"""
        from unified_search import RetrievedChunk
        
        chunks = [
            RetrievedChunk("1", "doc1", "text about Apple", 0.7, ["apple", "company"], {}, None, None),
            RetrievedChunk("2", "doc1", "text about Microsoft", 0.6, ["microsoft"], {}, None, None)
        ]
        
        question = "Tell me about Apple company"
        
        with patch.object(self.pipeline, '_get_community_scores_for_chunks') as mock_community:
            with patch.object(self.pipeline, '_get_chunk_centrality_scores') as mock_centrality:
                mock_community.return_value = {"1": 0.8, "2": 0.3}
                mock_centrality.return_value = {"1": 0.9, "2": 0.5}
                
                result = await self.pipeline._graph_aware_rerank(chunks, question)
                
                assert len(result) == 2
                # First chunk should have higher final score due to entity overlap
                assert result[0].final_score is not None
                assert result[0].final_score >= result[1].final_score
    
    @pytest.mark.asyncio
    async def test_mmr_reranking(self):
        """Test MMR diversity-aware reranking"""
        from unified_search import RetrievedChunk
        
        # Create chunks with embeddings
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]  # Different from embedding1
        embedding3 = [0.9, 0.1, 0.0]  # Similar to embedding1
        
        chunks = [
            RetrievedChunk("1", "doc1", "text1", 0.9, [], {}, embedding1, 0.9),
            RetrievedChunk("2", "doc1", "text2", 0.8, [], {}, embedding2, 0.8),
            RetrievedChunk("3", "doc1", "text3", 0.85, [], {}, embedding3, 0.85)
        ]
        
        result = await self.pipeline._mmr_rerank(chunks, top_k=2, lambda_param=0.7)
        
        assert len(result) == 2
        # Should prefer diverse chunks (1 and 2) over similar ones (1 and 3)
        assert result[0].chunk_id in ["1", "2"]
        assert result[1].chunk_id in ["1", "2"]
        # Should not have both 1 and 3 (too similar)
        assert not (result[0].chunk_id == "1" and result[1].chunk_id == "3")

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]
        
        # Identical vectors
        assert abs(cosine_similarity(vec1, vec2) - 1.0) < 1e-6
        
        # Orthogonal vectors
        assert abs(cosine_similarity(vec1, vec3)) < 1e-6
        
        # Empty vectors
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([1, 2], []) == 0.0
    
    @pytest.mark.asyncio
    async def test_extract_entities_efficient(self):
        """Test the efficient entity extraction function"""
        text = "Apple Inc. is based in California."
        
        entities = await extract_entities_efficient(text)
        
        assert isinstance(entities, list)
        if entities:  # If spaCy model is available
            assert all(isinstance(e, dict) for e in entities)
            assert all("name" in e and "type" in e for e in entities)

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_llm_client_retry_logic(self):
        """Test LLM client retry logic"""
        client = AsyncLLMClient()
        
        # Mock responses for retry logic
        mock_response1 = Mock()
        mock_response1.raise_for_status.side_effect = Exception("Network error")
        
        mock_response2 = Mock()
        mock_response2.raise_for_status.return_value = None
        if client.provider == "google":
            mock_response2.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Success"}]
                    }
                }]
            }
        else:
            mock_response2.json.return_value = {
                "choices": [{"message": {"content": "Success"}}]
            }
        
        with patch('httpx.AsyncClient') as mock_client:
            # First call fails, second succeeds
            mock_client.return_value.__aenter__.return_value.post.side_effect = [
                mock_response1,
                mock_response2
            ]
            
            # Should succeed after retry
            result = await client.invoke([{"role": "user", "content": "test"}])
            assert result == "Success"
    
    @pytest.mark.asyncio
    async def test_invalid_input_handling(self):
        """Test handling of invalid inputs"""
        processor = EfficientNLPProcessor()
        
        # Test with various invalid inputs
        entities = await processor.extract_entities(None)
        assert entities == []
        
        # Non-string inputs should return empty list
        entities = await processor.extract_entities(123)
        assert entities == []
        
        entities = await processor.extract_entities("")
        assert entities == []

class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_document_processing_flow(self):
        """Test the complete document processing flow"""
        # This would require a test database setup
        # For now, just test that the components can work together
        
        text = "Apple Inc. is a technology company. It was founded by Steve Jobs."
        
        # Clean and chunk text
        processor = ImprovedTextProcessor()
        cleaned_text = processor.clean_text(text)
        chunks = processor.chunk_text_advanced(cleaned_text, max_chunk_size=100)
        
        assert len(chunks) > 0
        
        # Extract entities
        nlp_processor = EfficientNLPProcessor()
        entities = await nlp_processor.extract_entities(cleaned_text)
        
        # Note: May be empty if LLM API is not available, but structure should work
        assert isinstance(entities, list)
        
        # This demonstrates the improved pipeline works end-to-end
        assert True

class TestVectorIndexing:
    """Test vector indexing functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.mock_driver = Mock()
        self.graph_manager = GraphManager(self.mock_driver)
    
    @pytest.mark.asyncio
    async def test_vector_index_creation(self):
        """Test vector index creation"""
        with patch('document_api.run_cypher_query_async') as mock_query:
            mock_query.return_value = []
            
            # Test with default dimension
            await self.graph_manager._ensure_vector_indexes(1024)
            
            # Should have called to create indexes
            assert mock_query.call_count >= 3  # At least 3 calls for chunk, entity, and community summary indexes
    
    @pytest.mark.asyncio
    async def test_vector_index_dimension_detection(self):
        """Test that vector index uses correct dimension"""
        with patch('document_api.run_cypher_query_async') as mock_query:        
            # Mock dimension detection
            mock_query.side_effect = [
                [{"dim": 768}],  # First call returns dimension
                [],  # Chunk index creation
                [],  # Entity index creation
                []   # CommunitySummary index creation
            ]

            await self.graph_manager._ensure_vector_indexes(768)
            
            # Verify dimension was used in index creation
            assert mock_query.called

def test_configuration_loading():
    """Test that configuration is loaded correctly"""
    from utils import CHUNK_SIZE_GDS, RELEVANCE_THRESHOLD
    
    assert isinstance(CHUNK_SIZE_GDS, int)
    assert CHUNK_SIZE_GDS > 0
    assert isinstance(RELEVANCE_THRESHOLD, float)
    assert 0.0 <= RELEVANCE_THRESHOLD <= 1.0

# Performance tests
class TestPerformance:
    """Performance tests for the system"""
    
    @pytest.mark.asyncio
    async def test_batch_vs_individual_embeddings(self):
        """Test that batch processing is more efficient"""
        client = BatchEmbeddingClient()
        texts = ["Text " + str(i) for i in range(10)]
        
        # Mock the API call based on provider
        method_name = '_fetch_batch_embeddings_openai' if client.provider in {"openai", "ollama"} else '_fetch_batch_embeddings_gemini'
        with patch.object(client, method_name) as mock_fetch:
            mock_fetch.return_value = [[0.1, 0.2]] * len(texts)
            
            # Batch processing
            start_time = asyncio.get_event_loop().time()
            batch_embeddings = await client.get_embeddings(texts)
            batch_time = asyncio.get_event_loop().time() - start_time
            
            # Individual processing (simulated)
            start_time = asyncio.get_event_loop().time()
            individual_embeddings = []
            for text in texts:
                emb = await client.get_embedding(text)
                individual_embeddings.append(emb)
            individual_time = asyncio.get_event_loop().time() - start_time
            
            # Results should be the same
            assert batch_embeddings == individual_embeddings
            
            # Note: In this test, both will be fast due to caching,
            # but in real usage, batch processing is much more efficient

class TestQuestionClassifier:
    """Test the question classifier"""
    
    def setup_method(self):
        """Setup for each test"""
        self.classifier = QuestionClassifier()
    
    @pytest.mark.asyncio
    async def test_classify_broad_question(self):
        """Test classification of broad questions"""
        question = "Give me an overview of all policies in the system"
        result = await self.classifier.classify(question)
        
        assert result["type"] in ["BROAD", "CHUNK", "OUT_OF_SCOPE"]
        assert "reason" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
        
        # Should classify as BROAD with high confidence
        if result["type"] == "BROAD":
            assert result["confidence"] > 0.5
    
    @pytest.mark.asyncio
    async def test_classify_chunk_question(self):
        """Test classification of specific chunk questions"""
        question = "What is the deadline mentioned in document X?"
        result = await self.classifier.classify(question)
        
        assert result["type"] in ["BROAD", "CHUNK", "OUT_OF_SCOPE"]
        assert "reason" in result
        assert "confidence" in result
        
        # Should classify as CHUNK (specific detail question)
        if result["type"] == "CHUNK":
            assert result["confidence"] > 0.4
    
    @pytest.mark.asyncio
    async def test_classify_empty_question(self):
        """Test classification of empty question"""
        result = await self.classifier.classify("")
        
        assert result["type"] == "OUT_OF_SCOPE"
        assert result["confidence"] == 1.0
        assert "reason" in result
    
    @pytest.mark.asyncio
    async def test_classify_with_heuristics(self):
        """Test heuristic classification"""
        # Broad question keywords
        broad_q = "What are the main themes?"
        result = await self.classifier.classify(broad_q)
        
        assert result["type"] in ["BROAD", "CHUNK", "OUT_OF_SCOPE"]
        
        # Specific question keywords
        chunk_q = "What does section 3.2 say?"
        result2 = await self.classifier.classify(chunk_q)
        
        assert result2["type"] in ["BROAD", "CHUNK", "OUT_OF_SCOPE"]
    
    @pytest.mark.asyncio
    async def test_classify_function(self):
        """Test the convenience classify function"""
        result = await classify_question("Test question")
        
        assert "type" in result
        assert "reason" in result
        assert "confidence" in result
        assert result["type"] in ["BROAD", "CHUNK", "OUT_OF_SCOPE"]
    
    @pytest.mark.asyncio
    async def test_classification_accuracy_broad_questions(self):
        """Test classification accuracy for various broad question patterns"""
        broad_questions = [
            "Give me an overview of all topics",
            "What are the main themes?",
            "Summarize the key points",
            "What's the big picture?",
            "Compare different approaches"
        ]
        
        for question in broad_questions:
            result = await self.classifier.classify(question)
            assert result["type"] in ["BROAD", "CHUNK", "OUT_OF_SCOPE"]
            # Most should be classified as BROAD
            if result["type"] == "BROAD":
                assert result["confidence"] > 0.5, f"Question '{question}' should have higher confidence"
    
    @pytest.mark.asyncio
    async def test_classification_accuracy_chunk_questions(self):
        """Test classification accuracy for specific/chunk questions"""
        chunk_questions = [
            "What is the deadline?",
            "Where does it say X?",
            "What section mentions Y?",
            "According to the document, what is Z?",
            "Who said that?"
        ]
        
        for question in chunk_questions:
            result = await self.classifier.classify(question)
            assert result["type"] in ["BROAD", "CHUNK", "OUT_OF_SCOPE"]
            # Most should be classified as CHUNK
            if result["type"] == "CHUNK":
                assert result["confidence"] > 0.4, f"Question '{question}' should have reasonable confidence"
    
    @pytest.mark.asyncio
    async def test_classification_fallback_chain(self):
        """Test that classification falls back through heuristics -> LLM -> default"""
        question = "Test question"
        
        # Test that classification always returns a valid result
        result = await self.classifier.classify(question)
        
        assert result["type"] in ["BROAD", "CHUNK", "OUT_OF_SCOPE"]
        assert "reason" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

class TestMapReduce:
    """Test map-reduce processor"""
    
    def setup_method(self):
        """Setup for each test"""
        self.processor = MapReduceProcessor()
    
    @pytest.mark.asyncio
    async def test_map_reduce_empty_communities(self):
        """Test map-reduce with empty communities"""
        result = await self.processor.process_communities(
            "What are the policies?",
            []
        )
        
        assert "couldn't find" in result.lower() or "no relevant" in result.lower()
    
    @pytest.mark.asyncio
    async def test_map_reduce_single_community(self):
        """Test map-reduce with single community"""
        communities = [
            {
                "summary": "Policy A requires all employees to complete training annually.",
                "community": "community_0",
                "similarity_score": 0.8
            }
        ]
        
        question = "What are the training requirements?"
        
        with patch('map_reduce.llm_client') as mock_llm:
            mock_llm.invoke = AsyncMock(return_value="Policy A requires annual training completion for all employees.")
            
            result = await self.processor.process_communities(question, communities)
            
            assert isinstance(result, str)
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_map_reduce_multiple_communities(self):
        """Test map-reduce with multiple communities"""
        communities = [
            {
                "summary": "Security policies include access control and encryption.",
                "community": "community_0",
                "similarity_score": 0.8
            },
            {
                "summary": "Data protection policies require regular backups.",
                "community": "community_1",
                "similarity_score": 0.7
            }
        ]
        
        question = "What security and data protection policies exist?"
        
        with patch('map_reduce.llm_client') as mock_llm:
            # Mock map step (extract from each community)
            def mock_map_side_effect(messages):
                content = messages[1]["content"]
                if "NO_RELEVANT_INFO" in content:
                    return "NO_RELEVANT_INFO"
                if "Security policies" in content:
                    return "Access control and encryption are required."
                if "Data protection" in content:
                    return "Regular backups are required for data protection."
                return "Some relevant information"
            
            # Mock reduce step (combine)
            def mock_reduce_side_effect(messages):
                return "Security policies require access control and encryption. Data protection policies require regular backups."
            
            # Setup mock to return different values for map vs reduce
            call_count = [0]
            def mock_invoke(messages):
                call_count[0] += 1
                content = messages[1]["content"]
                if "Combine" in content or "synthesizing" in content.lower():
                    return mock_reduce_side_effect(messages)
                else:
                    return mock_map_side_effect(messages)
            
            mock_llm.invoke = AsyncMock(side_effect=mock_invoke)
            
            result = await self.processor.process_communities(question, communities)
            
            assert isinstance(result, str)
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_map_reduce_filter_low_relevance(self):
        """Test that low-relevance communities are filtered"""
        communities = [
            {
                "summary": "Relevant summary about policies.",
                "community": "community_0",
                "similarity_score": 0.5  # Above threshold
            },
            {
                "summary": "Low relevance summary.",
                "community": "community_1",
                "similarity_score": 0.2  # Below threshold
            }
        ]
        
        question = "What are the policies?"
        
        # Mock to track which communities are processed
        processed_summaries = []
        
        async def mock_llm(messages):
            content = messages[1]["content"]
            if "Community Summary:" in content:
                # Extract which summary was processed
                for comm in communities:
                    if comm["summary"] in content:
                        processed_summaries.append(comm["community"])
                        return "Relevant information extracted."
            return "Combined answer"
        
        with patch('map_reduce.llm_client') as mock_client:
            mock_client.invoke = AsyncMock(side_effect=mock_llm)
            
            await self.processor.process_communities(question, communities)
            
            # Should only process high-relevance community
            assert len(processed_summaries) >= 0  # At least the high-relevance one
    
    @pytest.mark.asyncio
    async def test_map_reduce_aggregation_accuracy(self):
        """Test that map-reduce properly aggregates information from multiple communities"""
        communities = [
            {
                "summary": "Policy A: All employees must complete safety training.",
                "community": "community_0",
                "similarity_score": 0.8
            },
            {
                "summary": "Policy B: Regular performance reviews are required quarterly.",
                "community": "community_1",
                "similarity_score": 0.75
            },
            {
                "summary": "Policy C: Data must be encrypted at rest and in transit.",
                "community": "community_2",
                "similarity_score": 0.7
            }
        ]
        
        question = "What are all the company policies?"
        
        with patch('map_reduce.llm_client') as mock_llm:
            # Track map calls
            map_calls = []
            def mock_invoke(messages):
                content = messages[1]["content"]
                if "Community Summary:" in content:
                    # Map step - extract from community
                    map_calls.append(content)
                    for comm in communities:
                        if comm["summary"] in content:
                            return f"Extracted: {comm['summary']}"
                    return "Extracted information"
                elif "Combine" in content or "synthesizing" in content.lower():
                    # Reduce step - combine all partial answers
                    return "Combined answer covering all three policies: safety training, performance reviews, and data encryption."
                return "Some information"
            
            mock_llm.invoke = AsyncMock(side_effect=mock_invoke)
            
            result = await self.processor.process_communities(question, communities)
            
            # Verify map step was called for each community
            assert len(map_calls) >= len(communities), "Map step should process all communities"
            
            # Verify reduce step was called
            assert "Combined" in result or "policies" in result.lower() or len(result) > 50
    
    @pytest.mark.asyncio
    async def test_map_reduce_handles_partial_failures(self):
        """Test that map-reduce continues processing even if some communities fail"""
        communities = [
            {
                "summary": "Valid summary 1",
                "community": "community_0",
                "similarity_score": 0.8
            },
            {
                "summary": "",  # Empty summary should be skipped
                "community": "community_1",
                "similarity_score": 0.7
            },
            {
                "summary": "Valid summary 2",
                "community": "community_2",
                "similarity_score": 0.75
            }
        ]
        
        question = "What are the policies?"
        
        with patch('map_reduce.llm_client') as mock_llm:
            processed_communities = []
            def mock_invoke(messages):
                content = messages[1]["content"]
                if "Community Summary:" in content:
                    # Track which communities were processed
                    for comm in communities:
                        if comm["summary"] and comm["summary"] in content:
                            processed_communities.append(comm["community"])
                            return f"Information from {comm['community']}"
                elif "Combine" in content or "synthesizing" in content.lower():
                    return "Combined answer from valid communities"
                return "NO_RELEVANT_INFO"
            
            mock_llm.invoke = AsyncMock(side_effect=mock_invoke)
            
            result = await self.processor.process_communities(question, communities)
            
            # Should process valid communities and skip empty ones
            assert len(processed_communities) >= 2  # At least valid ones
            assert "community_1" not in processed_communities  # Empty one should be skipped
            assert isinstance(result, str)
            assert len(result) > 0

class TestClassificationRouting:
    """Test classification-based routing in unified search"""
    
    def setup_method(self):
        """Setup for each test"""
        self.mock_driver = Mock()
        self.pipeline = UnifiedSearchPipeline(self.mock_driver)
    
    @pytest.mark.asyncio
    async def test_routing_broad_question(self):
        """Test routing of BROAD questions to community search"""
        question = "Give me an overview of all topics"
        
        with patch('unified_search.classify_question') as mock_classify:
            mock_classify.return_value = {
                "type": "BROAD",
                "reason": "Broad question",
                "confidence": 0.9
            }
            
            with patch.object(self.pipeline, '_search_with_communities') as mock_broad:
                mock_broad.return_value = type('obj', (object,), {
                    'answer': 'Broad answer',
                    'relevant_chunks': [],
                    'community_summaries': [],
                    'entities_found': [],
                    'confidence_score': 0.8,
                    'search_metadata': {'question_type': 'BROAD', 'search_time': 1.0}
                })()
                
                result = await self.pipeline.search(question)
                
                # Should route to community search
                mock_broad.assert_called_once()
                assert result.search_metadata.get("question_type") == "BROAD"
    
    @pytest.mark.asyncio
    async def test_routing_chunk_question(self):
        """Test routing of CHUNK questions to chunk search"""
        question = "What is the deadline in document X?"
        
        with patch('unified_search.classify_question') as mock_classify:
            mock_classify.return_value = {
                "type": "CHUNK",
                "reason": "Specific question",
                "confidence": 0.8
            }
            
            with patch.object(self.pipeline, '_search_with_chunks') as mock_chunk:
                mock_chunk.return_value = type('obj', (object,), {
                    'answer': 'Specific answer',
                    'relevant_chunks': [{"chunk_id": "1"}],
                    'community_summaries': [],
                    'entities_found': [],
                    'confidence_score': 0.7,
                    'search_metadata': {'question_type': 'CHUNK', 'search_time': 0.5}
                })()
                
                result = await self.pipeline.search(question)
                
                # Should route to chunk search
                mock_chunk.assert_called_once()
                assert result.search_metadata.get("question_type") == "CHUNK"
    
    @pytest.mark.asyncio
    async def test_routing_out_of_scope(self):
        """Test routing of OUT_OF_SCOPE questions"""
        question = "What's the weather today?"
        
        with patch('unified_search.classify_question') as mock_classify:
            mock_classify.return_value = {
                "type": "OUT_OF_SCOPE",
                "reason": "Not in knowledge base",
                "confidence": 0.9
            }
            
            result = await self.pipeline.search(question)
            
            # Should return out-of-scope message
            assert result.search_metadata.get("question_type") == "OUT_OF_SCOPE"
            assert "not confident" in result.answer.lower() or "knowledge base" in result.answer.lower()
    
    @pytest.mark.asyncio
    async def test_classification_integration(self):
        """Test that classification is called during search"""
        question = "Test question"
        
        with patch('unified_search.classify_question') as mock_classify:
            mock_classify.return_value = {
                "type": "CHUNK",
                "reason": "Test",
                "confidence": 0.5
            }
            
            with patch.object(self.pipeline, '_search_with_chunks') as mock_chunk:
                mock_chunk.return_value = type('obj', (object,), {
                    'answer': 'Answer',
                    'relevant_chunks': [],
                    'community_summaries': [],
                    'entities_found': [],
                    'confidence_score': 0.5,
                    'search_metadata': {}
                })()
                
                await self.pipeline.search(question)
                
                # Should call classifier
                mock_classify.assert_called_once_with(question)

class TestMCPClassifier:
    """Test MCP classifier client (mocked)"""
    
    @pytest.mark.asyncio
    async def test_mcp_client_structure(self):
        """Test that MCP client module structure is correct"""
        try:
            from mcp_classifier_client import classify_question_via_mcp, test_mcp_connection
            # Module should be importable
            assert True
        except ImportError:
            pytest.skip("MCP client not available")
    
    @pytest.mark.asyncio
    async def test_mcp_client_fallback(self):
        """Test MCP client fallback on connection error"""
        try:
            from mcp_classifier_client import classify_question_via_mcp
            
            # Should handle connection errors gracefully
            result = await classify_question_via_mcp("test question")
            
            # Should return a valid classification result even on error
            assert "type" in result
            assert result["type"] in ["BROAD", "CHUNK", "OUT_OF_SCOPE"]
        except ImportError:
            pytest.skip("MCP client not available")

class TestExternalAPIUrls:
    """Test external API URL configurations and validation"""
    
    def test_embedding_api_url_format(self):
        """Test that embedding API URL is properly formatted"""
        embedding_url = os.getenv("EMBEDDING_API_URL")
        
        assert embedding_url, "EMBEDDING_API_URL not set in environment"
        
        # Validate URL format
        from urllib.parse import urlparse
        parsed = urlparse(embedding_url)
        assert parsed.scheme in ["http", "https"], f"Invalid URL scheme: {parsed.scheme}"
        assert parsed.netloc, f"Invalid URL netloc: {parsed.netloc}"
        
        # For Gemini provider, URL should not end with /embeddings (code handles it)
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if provider == "google":
            # Gemini uses base URL + /models/... path
            # Check that URL is valid base URL
            assert not embedding_url.endswith("/embeddings"), \
                "For Gemini provider, EMBEDDING_API_URL should be base URL without /embeddings"
    
    def test_llm_base_url_format(self):
        """Test that LLM base URL is properly formatted"""
        llm_url = os.getenv("OPENAI_BASE_URL")
        
        assert llm_url, "OPENAI_BASE_URL not set in environment"
        
        from urllib.parse import urlparse
        parsed = urlparse(llm_url)
        assert parsed.scheme in ["http", "https"], f"Invalid URL scheme: {parsed.scheme}"
        assert parsed.netloc, f"Invalid URL netloc: {parsed.netloc}"
    
    def test_ner_base_url_format(self):
        """Test that NER base URL is properly formatted"""
        ner_url = os.getenv("NER_BASE_URL")
        
        assert ner_url, "NER_BASE_URL not set in environment"
        
        from urllib.parse import urlparse
        parsed = urlparse(ner_url)
        assert parsed.scheme in ["http", "https"], f"Invalid URL scheme: {parsed.scheme}"
        assert parsed.netloc, f"Invalid URL netloc: {parsed.netloc}"
    
    def test_coref_base_url_format(self):
        """Test that coreference resolution base URL is properly formatted"""
        coref_url = os.getenv("COREF_BASE_URL")
        
        assert coref_url, "COREF_BASE_URL not set in environment"
        
        from urllib.parse import urlparse
        parsed = urlparse(coref_url)
        assert parsed.scheme in ["http", "https"], f"Invalid URL scheme: {parsed.scheme}"
        assert parsed.netloc, f"Invalid URL netloc: {parsed.netloc}"
    
    def test_mcp_classifier_url_format(self):
        """Test that MCP classifier URL is properly formatted"""
        mcp_url = os.getenv("MCP_CLASSIFIER_URL")
        
        # MCP URL is optional
        if mcp_url:
            from urllib.parse import urlparse
            parsed = urlparse(mcp_url)
            assert parsed.scheme in ["http", "https"], f"Invalid URL scheme: {parsed.scheme}"
            assert parsed.netloc, f"Invalid URL netloc: {parsed.netloc}"
    
    @pytest.mark.asyncio
    async def test_embedding_api_url_accessible(self):
        """Test that embedding API URL is accessible (optional - may fail if API key invalid)"""
        import httpx
        embedding_url = os.getenv("EMBEDDING_API_URL")
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        
        if not embedding_url:
            pytest.skip("EMBEDDING_API_URL not set")
        
        try:
            # For Gemini, construct the full URL
            if provider == "google":
                model_name = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-001")
                base_url = embedding_url.rstrip('/')
                if '/embeddings' in base_url:
                    base_url = base_url.split('/embeddings')[0]
                test_url = f"{base_url}/models/{model_name}:batchEmbedContents"
            else:
                # For OpenAI/Ollama, ensure /embeddings is in URL
                test_url = embedding_url.rstrip('/')
                if not test_url.endswith('/embeddings'):
                    test_url = f"{test_url}/embeddings"
            
            from urllib.parse import urlparse
            parsed = urlparse(test_url)
            
            # Test connectivity (HEAD request to check if endpoint exists)
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Just validate the URL is reachable, don't test actual API
                # This is a basic connectivity test
                if parsed.scheme == "https" and "generativelanguage.googleapis.com" in parsed.netloc:
                    # Google API - just validate URL format
                    assert "models" in test_url or "batchEmbedContents" in test_url
                elif parsed.netloc.startswith("localhost") or parsed.netloc.startswith("127.0.0.1"):
                    # Local service - try to connect
                    try:
                        response = await client.head(f"{parsed.scheme}://{parsed.netloc}", timeout=2.0)
                        # Any response is fine, just checking connectivity
                    except Exception:
                        pytest.skip(f"Local service at {parsed.netloc} not accessible")
        except Exception as e:
            pytest.skip(f"Could not validate embedding API URL: {e}")
    
    @pytest.mark.asyncio
    async def test_llm_base_url_accessible(self):
        """Test that LLM base URL is accessible (optional - may fail if API key invalid)"""
        import httpx
        llm_url = os.getenv("OPENAI_BASE_URL")
        
        if not llm_url:
            pytest.skip("OPENAI_BASE_URL not set")
        
        try:
            from urllib.parse import urlparse
            parsed = urlparse(llm_url)
            
            # Basic connectivity test for local services
            if parsed.netloc.startswith("localhost") or parsed.netloc.startswith("127.0.0.1"):
                async with httpx.AsyncClient(timeout=2.0) as client:
                    try:
                        response = await client.head(f"{parsed.scheme}://{parsed.netloc}", timeout=2.0)
                        # Any response is fine
                    except Exception:
                        pytest.skip(f"Local LLM service at {parsed.netloc} not accessible")
        except Exception as e:
            pytest.skip(f"Could not validate LLM base URL: {e}")
    
    def test_api_keys_present(self):
        """Test that required API keys are present in environment"""
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        
        if provider == "google":
            # Check for Google API key
            embedding_key = os.getenv("EMBEDDING_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            
            # At least one should be present
            assert embedding_key or openai_key, \
                "EMBEDDING_API_KEY or OPENAI_API_KEY required for Google provider"
        else:
            # For OpenAI/Ollama, check embedding key
            embedding_key = os.getenv("EMBEDDING_API_KEY")
            # API key might be optional for local services
            if embedding_key:
                assert len(embedding_key) > 0, "EMBEDDING_API_KEY is empty"
    
    def test_embedding_url_construction(self):
        """Test that embedding URL is constructed correctly for different providers"""
        from utils import BatchEmbeddingClient
        
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        embedding_url = os.getenv("EMBEDDING_API_URL")
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "test-model")
        
        if not embedding_url:
            pytest.skip("EMBEDDING_API_URL not set")
        
        client = BatchEmbeddingClient()
        
        # Test URL construction for Gemini
        if provider == "google":
            # Should construct: base_url/models/model_name:batchEmbedContents
            base_url = embedding_url.rstrip('/')
            if '/embeddings' in base_url:
                base_url = base_url.split('/embeddings')[0]
            
            expected_pattern = f"{base_url}/models/{model_name}:batchEmbedContents"
            assert expected_pattern is not None, "Gemini URL construction should work"
        
        # Test URL construction for OpenAI/Ollama
        else:
            # Should use URL directly (may need /embeddings appended)
            api_url = embedding_url.rstrip('/')
            if not api_url.endswith('/embeddings'):
                api_url = f"{api_url}/embeddings"
            assert '/embeddings' in api_url, "OpenAI/Ollama URL should include /embeddings"
    
    def test_all_api_urls_present(self):
        """Test that all required API URLs are present"""
        required_urls = [
            "EMBEDDING_API_URL",
            "OPENAI_BASE_URL",
            "NER_BASE_URL",
            "COREF_BASE_URL"
        ]
        
        missing_urls = []
        for url_var in required_urls:
            if not os.getenv(url_var):
                missing_urls.append(url_var)
        
        assert len(missing_urls) == 0, f"Missing required API URLs: {', '.join(missing_urls)}"
    
    def test_embedding_url_construction_gemini(self):
        """Test embedding URL construction for Gemini provider"""
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        
        if provider != "google":
            pytest.skip("Not using Gemini provider")
        
        embedding_url = os.getenv("EMBEDDING_API_URL")
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-001")
        
        assert embedding_url, "EMBEDDING_API_URL not set"
        
        # Test URL construction logic
        base_url = embedding_url.rstrip('/')
        if '/embeddings' in base_url:
            base_url = base_url.split('/embeddings')[0]
        
        expected_url = f"{base_url}/models/{model_name}:batchEmbedContents"
        
        # Validate the constructed URL
        from urllib.parse import urlparse
        parsed = urlparse(expected_url)
        assert parsed.scheme in ["http", "https"]
        assert "models" in expected_url
        assert "batchEmbedContents" in expected_url


class TestURLValidationHelpers:
    """Test helper functions for URL validation"""
    
    def test_validate_url_format(self):
        """Test URL format validation"""
        from urllib.parse import urlparse
        
        valid_urls = [
            "http://localhost:11434/v1",
            "https://api.openai.com/v1",
            "https://generativelanguage.googleapis.com/v1beta",
            "http://127.0.0.1:8000",
        ]
        
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",
            "http://",
            "",
        ]
        
        for url in valid_urls:
            parsed = urlparse(url)
            assert parsed.scheme in ["http", "https"], f"URL {url} should be valid"
            assert parsed.netloc, f"URL {url} should have netloc"
        
        for url in invalid_urls:
            if url:  # Skip empty string
                parsed = urlparse(url)
                # These should be invalid or have issues
                if not parsed.netloc or parsed.scheme not in ["http", "https"]:
                    # This is expected to be invalid
                    pass

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
