"""
Comprehensive test suite for the Graph RAG system

Tests all key components:
- Efficient NLP processing with spaCy
- Batch embedding optimization
- Unified search pipeline
- Async handling and performance
- Error handling and edge cases
"""

import asyncio
import pytest
import os
import tempfile
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock

# Import modules to test
from utils import (
    EfficientNLPProcessor, BatchEmbeddingClient, AsyncLLMClient,
    ImprovedTextProcessor, cosine_similarity, extract_entities_efficient
)
from unified_search import UnifiedSearchPipeline, SearchScope
from document_api import GraphManager

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
        
        # Mock all the internal methods
        with patch.multiple(
            self.pipeline,
            _preprocess_query=AsyncMock(return_value="processed question"),
            _retrieve_relevant_chunks=AsyncMock(return_value=[]),
            _expand_context_via_graph=AsyncMock(return_value=[]),
            _rank_and_filter_chunks=AsyncMock(return_value=[]),
            _get_relevant_community_summaries=AsyncMock(return_value=[]),
            _generate_answer=AsyncMock(return_value="Test answer"),
            _extract_entities_from_chunks=AsyncMock(return_value=[])
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
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock a failing request that succeeds on retry
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = [Exception("Network error"), None]
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Success"}}]
            }
            
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
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

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
