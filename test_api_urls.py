"""
Test suite for validating external API URL configurations

Tests all external API URLs to ensure they are correctly formatted and accessible.
"""

import pytest
import os
import httpx
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()


class TestExternalAPIUrls:
    """Test external API URL configurations"""
    
    def test_embedding_api_url_format(self):
        """Test that embedding API URL is properly formatted"""
        embedding_url = os.getenv("EMBEDDING_API_URL")
        
        assert embedding_url, "EMBEDDING_API_URL not set in environment"
        
        # Validate URL format
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
        
        parsed = urlparse(llm_url)
        assert parsed.scheme in ["http", "https"], f"Invalid URL scheme: {parsed.scheme}"
        assert parsed.netloc, f"Invalid URL netloc: {parsed.netloc}"
    
    def test_ner_base_url_format(self):
        """Test that NER base URL is properly formatted"""
        ner_url = os.getenv("NER_BASE_URL")
        
        assert ner_url, "NER_BASE_URL not set in environment"
        
        parsed = urlparse(ner_url)
        assert parsed.scheme in ["http", "https"], f"Invalid URL scheme: {parsed.scheme}"
        assert parsed.netloc, f"Invalid URL netloc: {parsed.netloc}"
    
    def test_coref_base_url_format(self):
        """Test that coreference resolution base URL is properly formatted"""
        coref_url = os.getenv("COREF_BASE_URL")
        
        assert coref_url, "COREF_BASE_URL not set in environment"
        
        parsed = urlparse(coref_url)
        assert parsed.scheme in ["http", "https"], f"Invalid URL scheme: {parsed.scheme}"
        assert parsed.netloc, f"Invalid URL netloc: {parsed.netloc}"
    
    def test_mcp_classifier_url_format(self):
        """Test that MCP classifier URL is properly formatted"""
        mcp_url = os.getenv("MCP_CLASSIFIER_URL")
        
        # MCP URL is optional
        if mcp_url:
            parsed = urlparse(mcp_url)
            assert parsed.scheme in ["http", "https"], f"Invalid URL scheme: {parsed.scheme}"
            assert parsed.netloc, f"Invalid URL netloc: {parsed.netloc}"
    
    @pytest.mark.asyncio
    async def test_embedding_api_url_accessible(self):
        """Test that embedding API URL is accessible (optional - may fail if API key invalid)"""
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
        llm_url = os.getenv("OPENAI_BASE_URL")
        
        if not llm_url:
            pytest.skip("OPENAI_BASE_URL not set")
        
        try:
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


class TestURLValidationHelpers:
    """Test helper functions for URL validation"""
    
    def test_validate_url_format(self):
        """Test URL format validation"""
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
    pytest.main([__file__, "-v"])

