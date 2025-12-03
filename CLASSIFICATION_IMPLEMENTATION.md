# Question Classification & Map-Reduce Implementation

## Overview

This document describes the implementation of agentic LLM-based question classification and intelligent routing in the Graph RAG system. The system now automatically classifies questions and routes them to the most appropriate search strategy.

## Architecture

### Question Classification Flow

```
User Question
    ↓
Question Classifier (LLM + Heuristics)
    ↓
Classification Result
    ├── OUT_OF_SCOPE → Polite fallback
    ├── BROAD → Community Summaries + Map-Reduce
    └── CHUNK → Chunk-level Retrieval
```

### Implementation Components

1. **Question Classifier** (`question_classifier.py`)
   - LLM-based classification using the same LLM client
   - Heuristic fallback for fast classification
   - Returns classification type, reason, and confidence

2. **Map-Reduce Processor** (`map_reduce.py`)
   - Map step: Extract relevant info from each community summary
   - Reduce step: Combine partial answers into comprehensive answer
   - Handles large sets of community summaries efficiently

3. **Unified Search Pipeline** (Updated `unified_search.py`)
   - Integrated classification at the start of search
   - Routes to appropriate search strategy based on classification
   - Maintains backward compatibility

## Question Types

### BROAD
**Use Case**: Questions requiring broad understanding or overviews
- "Give me an overview of all policies"
- "What are the main themes in the documents?"
- "Compare different approaches"
- **Strategy**: Community summaries with map-reduce

### CHUNK
**Use Case**: Questions requiring specific details
- "What is the deadline mentioned in document X?"
- "According to the document, what did the CEO say?"
- "Which section discusses this topic?"
- **Strategy**: Chunk-level vector similarity search

### OUT_OF_SCOPE
**Use Case**: Questions not covered by knowledge base
- Personal questions
- Real-time events
- Unrelated topics
- **Strategy**: Polite fallback message

## Map-Reduce Process

### Map Step
For each community summary:
1. Extract relevant information related to the question
2. Generate a partial answer from that community
3. Filter out irrelevant communities (NO_RELEVANT_INFO)

### Reduce Step
1. Combine all partial answers
2. Remove redundancy
3. Resolve conflicts
4. Synthesize into comprehensive final answer

### Example Flow

```
Question: "What are the main security policies?"

Map Step:
- Community 1 (Access Control): "Policies include role-based access..."
- Community 2 (Data Protection): "Data encryption and backup policies..."
- Community 3 (Network Security): "Firewall and intrusion detection..."

Reduce Step:
- Combines all three into comprehensive answer about security policies
```

## Configuration

### Environment Variables

```bash
# Question Classifier
CLASSIFIER_USE_HEURISTICS=true      # Enable heuristic classification
CLASSIFIER_USE_LLM=true             # Enable LLM-based classification

# Map-Reduce for Broad Questions
MAP_REDUCE_MAX_COMMUNITIES=50       # Maximum communities to process
MAP_REDUCE_BATCH_SIZE=5             # Batch size for parallel processing
MAP_REDUCE_MIN_RELEVANCE=0.3        # Minimum relevance threshold
BROAD_SEARCH_MAX_COMMUNITIES=20     # Max communities for broad search
```

## API Usage

### Standard Search (Auto-Classifies)

```http
POST /api/search/search
Content-Type: application/json

{
  "question": "Give me an overview of all policies",
  "conversation_history": "Previous context...",
  "doc_id": "optional-doc-id",
  "scope": "hybrid",
  "max_chunks": 7
}
```

**Response includes:**
- Classification metadata in `search_metadata.question_type`
- Strategy used (`map_reduce_communities` or `chunk_level_retrieval`)
- Processing details

### Response Structure

```json
{
  "answer": "...",
  "confidence_score": 0.85,
  "chunks_used": 0,  // 0 for broad questions
  "entities_found": [...],
  "search_time": 2.34,
  "metadata": {
    "question_type": "BROAD",
    "strategy": "map_reduce_communities",
    "communities_used": 12,
    "processing_details": {...}
  }
}
```

## Performance Considerations

### Classification
- **Heuristics**: ~1ms (instant)
- **LLM**: ~200-500ms (depending on LLM latency)
- Combined approach balances speed and accuracy

### Map-Reduce
- **Map Step**: Parallel processing of communities (batch size: 5)
- **Reduce Step**: Single LLM call to combine results
- **Total Time**: ~2-5 seconds for 20 communities

### Optimization Tips
1. Use heuristics for fast classification (set `CLASSIFIER_USE_LLM=false`)
2. Reduce `BROAD_SEARCH_MAX_COMMUNITIES` for faster broad searches
3. Increase `MAP_REDUCE_BATCH_SIZE` for better parallelization
4. Adjust `MAP_REDUCE_MIN_RELEVANCE` to filter low-quality communities

## Benefits

1. **Intelligent Routing**: Automatically selects the best search strategy
2. **Efficiency**: Broad questions use summaries (faster), specific questions use chunks (precise)
3. **Scalability**: Map-reduce handles large numbers of communities
4. **Flexibility**: Heuristic + LLM classification balances speed and accuracy
5. **Backward Compatible**: Existing code continues to work

## Future Enhancements

1. **Streaming Responses**: Stream map-reduce results
2. **Caching**: Cache classification results for similar questions
3. **Fine-tuning**: Fine-tune classifier on domain-specific questions
4. **Multi-language**: Extend classification to other languages
5. **Custom Classification Types**: Add domain-specific question types

## Testing

### Test Classification

```python
from question_classifier import classify_question

# Test broad question
result = await classify_question("Give me an overview of all policies")
assert result["type"] == "BROAD"

# Test chunk question
result = await classify_question("What is the deadline in document X?")
assert result["type"] == "CHUNK"
```

### Test Map-Reduce

```python
from map_reduce import map_reduce_communities

communities = [
    {"summary": "...", "community": "policy1", "similarity_score": 0.8},
    {"summary": "...", "community": "policy2", "similarity_score": 0.7}
]

answer = await map_reduce_communities("What are the policies?", communities)
```

## Troubleshooting

### Classification Always Returns CHUNK
- Check if `CLASSIFIER_USE_LLM=true`
- Verify LLM is accessible
- Check logs for classification errors

### Map-Reduce Too Slow
- Reduce `BROAD_SEARCH_MAX_COMMUNITIES`
- Increase `MAP_REDUCE_BATCH_SIZE`
- Check LLM latency

### No Communities Found
- Verify community summaries exist in database
- Check `MAP_REDUCE_MIN_RELEVANCE` threshold
- Ensure documents have been processed with community detection

## Files Modified/Created

- ✅ `question_classifier.py` - New classification module
- ✅ `map_reduce.py` - New map-reduce processor
- ✅ `unified_search.py` - Updated with classification routing
- ✅ `search_api.py` - Updated API description
- ✅ `.env` - Added configuration options

## Summary

The system now intelligently classifies questions and routes them to the optimal search strategy:
- **Broad questions** → Community summaries with map-reduce for comprehensive answers
- **Specific questions** → Chunk-level retrieval for precise answers
- **Automatic** → No manual configuration needed

This provides better answers, faster responses, and more efficient resource usage.

