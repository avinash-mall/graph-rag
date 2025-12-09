# API URL Fixes and Validation Tests

## Summary

Fixed the embedding API URL construction issue and added comprehensive tests to validate all external API URLs used in the application.

## Issues Fixed

### 1. Embedding API URL Construction ✅

**Problem**: The embedding API URL was incorrectly constructed for Gemini provider:
- The `.env` file had: `EMBEDDING_API_URL="https://generativelanguage.googleapis.com/v1beta/embeddings"`
- But Gemini embedding API needs: `https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents`
- The code was appending `/models/...` to a URL that already contained `/embeddings`, resulting in an incorrect endpoint

**Solution**:
1. Updated `.env` file to use base URL without `/embeddings` for Gemini
2. Fixed URL construction logic in `utils.py` to:
   - For Gemini: Remove `/embeddings` if present, then append `/models/{model}:batchEmbedContents`
   - For OpenAI/Ollama: Ensure URL ends with `/embeddings`

**Code Changes** (`utils.py`):

```python
# Gemini embedding URL construction
base_url = self.embedding_api_url.rstrip('/')
if '/embeddings' in base_url:
    base_url = base_url.split('/embeddings')[0]  # Remove '/embeddings'
api_url = f"{base_url}/models/{self.model_name}:batchEmbedContents"

# OpenAI/Ollama embedding URL construction  
api_url = self.embedding_api_url.rstrip('/')
if not api_url.endswith('/embeddings'):
    api_url = f"{api_url}/embeddings"
```

### 2. Environment Configuration ✅

**Updated `.env` file**:
- Changed `EMBEDDING_API_URL` from `https://generativelanguage.googleapis.com/v1beta/embeddings` 
  to `https://generativelanguage.googleapis.com/v1beta`
- Added clarifying comment about URL format for different providers

## Tests Added

### New Test Module: `test_api_urls.py`

Comprehensive test suite with 10 tests covering:

1. **URL Format Validation**:
   - `test_embedding_api_url_format` - Validates embedding API URL format
   - `test_llm_base_url_format` - Validates LLM base URL format
   - `test_ner_base_url_format` - Validates NER API URL format
   - `test_coref_base_url_format` - Validates coreference API URL format
   - `test_mcp_classifier_url_format` - Validates MCP classifier URL format

2. **URL Accessibility** (optional, skips on failure):
   - `test_embedding_api_url_accessible` - Tests embedding API connectivity
   - `test_llm_base_url_accessible` - Tests LLM API connectivity

3. **Configuration Validation**:
   - `test_api_keys_present` - Validates required API keys are present
   - `test_embedding_url_construction` - Tests URL construction logic

4. **Helper Functions**:
   - `test_validate_url_format` - Tests URL format validation helpers

### Updated `test_system.py`

Added `TestExternalAPIUrls` class with 6 tests:

1. `test_embedding_api_url_format` - Validates embedding URL format
2. `test_llm_base_url_format` - Validates LLM URL format
3. `test_ner_base_url_format` - Validates NER URL format
4. `test_coref_base_url_format` - Validates coreference URL format
5. `test_all_api_urls_present` - Ensures all required URLs are configured
6. `test_embedding_url_construction_gemini` - Tests Gemini-specific URL construction

## Test Results

✅ **All 16 tests passing** (10 from `test_api_urls.py` + 6 from `test_system.py`)

```
test_api_urls.py::TestExternalAPIUrls::test_embedding_api_url_format PASSED
test_api_urls.py::TestExternalAPIUrls::test_llm_base_url_format PASSED
test_api_urls.py::TestExternalAPIUrls::test_ner_base_url_format PASSED
test_api_urls.py::TestExternalAPIUrls::test_coref_base_url_format PASSED
test_api_urls.py::TestExternalAPIUrls::test_mcp_classifier_url_format PASSED
test_api_urls.py::TestExternalAPIUrls::test_embedding_api_url_accessible PASSED
test_api_urls.py::TestExternalAPIUrls::test_llm_base_url_accessible PASSED
test_api_urls.py::TestExternalAPIUrls::test_api_keys_present PASSED
test_api_urls.py::TestExternalAPIUrls::test_embedding_url_construction PASSED
test_api_urls.py::TestURLValidationHelpers::test_validate_url_format PASSED
test_system.py::TestExternalAPIUrls::test_embedding_api_url_format PASSED
test_system.py::TestExternalAPIUrls::test_llm_base_url_format PASSED
test_system.py::TestExternalAPIUrls::test_ner_base_url_format PASSED
test_system.py::TestExternalAPIUrls::test_coref_base_url_format PASSED
test_system.py::TestExternalAPIUrls::test_all_api_urls_present PASSED
test_system.py::TestExternalAPIUrls::test_embedding_url_construction_gemini PASSED
```

## External API URLs Validated

The tests validate the following external API URLs:

1. **Embedding API**: `EMBEDDING_API_URL`
   - Gemini: `https://generativelanguage.googleapis.com/v1beta`
   - OpenAI/Ollama: `http://localhost:11434/v1/embeddings`

2. **LLM API**: `OPENAI_BASE_URL`
   - Example: `https://generativelanguage.googleapis.com/v1beta`

3. **NER API**: `NER_BASE_URL`
   - Example: `https://generativelanguage.googleapis.com/v1beta`

4. **Coreference API**: `COREF_BASE_URL`
   - Example: `https://generativelanguage.googleapis.com/v1beta`

5. **MCP Classifier** (optional): `MCP_CLASSIFIER_URL`
   - Example: `http://localhost:8001/mcp`

## Running the Tests

### Run all API URL tests:
```bash
python -m pytest test_api_urls.py -v
```

### Run API URL tests from test_system.py:
```bash
python -m pytest test_system.py::TestExternalAPIUrls -v
```

### Run all API URL validation tests:
```bash
python -m pytest test_api_urls.py test_system.py::TestExternalAPIUrls -v
```

## Configuration Notes

### For Gemini Provider:
```env
LLM_PROVIDER=google
EMBEDDING_API_URL="https://generativelanguage.googleapis.com/v1beta"
EMBEDDING_MODEL_NAME="gemini-embedding-001"
```

### For OpenAI/Ollama Provider:
```env
LLM_PROVIDER=openai
EMBEDDING_API_URL="http://localhost:11434/v1/embeddings"
EMBEDDING_MODEL_NAME="mxbai-embed-large"
```

## Benefits

1. ✅ **Correct URL Construction**: Embedding API URLs are now constructed correctly for both Gemini and OpenAI/Ollama providers
2. ✅ **Comprehensive Validation**: All external API URLs are validated for correct format
3. ✅ **Early Error Detection**: Tests catch configuration errors before runtime
4. ✅ **Documentation**: Tests serve as documentation for expected URL formats
5. ✅ **CI/CD Ready**: Tests can be run in CI/CD pipelines to validate configuration

## Files Modified

- ✅ `utils.py` - Fixed embedding API URL construction logic
- ✅ `.env` - Updated embedding API URL and added clarifying comments
- ✅ `test_api_urls.py` - New comprehensive test module
- ✅ `test_system.py` - Added API URL validation tests

## Next Steps

1. ✅ All fixes implemented
2. ✅ All tests passing
3. ✅ Configuration validated

The embedding API URL issue is now fixed, and comprehensive validation ensures all external API URLs are correctly configured.

