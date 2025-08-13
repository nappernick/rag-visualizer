"""
Comprehensive tests for Enhanced Hybrid Search with Reciprocal Rank Fusion.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import redis
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.core.retrieval.hybrid_search import (
    EnhancedHybridSearch,
    SearchConfig,
    RedisSearchIndex
)
from src.models import RetrievalResult, Chunk, Document


class TestSearchConfig:
    """Test SearchConfig dataclass and configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SearchConfig()
        
        assert config.vector_weight == 0.4
        assert config.keyword_weight == 0.3
        assert config.metadata_weight == 0.3
        assert config.vector_top_k == 20
        assert config.keyword_top_k == 20
        assert config.metadata_top_k == 10
        assert config.rrf_k == 60
        assert config.use_parallel == True
        assert config.keyword_boost_terms is None
        assert config.metadata_filters is None
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SearchConfig(
            vector_weight=0.5,
            keyword_weight=0.2,
            metadata_weight=0.3,
            vector_top_k=30,
            rrf_k=100,
            use_parallel=False,
            keyword_boost_terms=['important', 'critical'],
            metadata_filters={'doc_type': 'project'}
        )
        
        assert config.vector_weight == 0.5
        assert config.keyword_weight == 0.2
        assert config.vector_top_k == 30
        assert config.rrf_k == 100
        assert config.use_parallel == False
        assert 'important' in config.keyword_boost_terms
        assert config.metadata_filters['doc_type'] == 'project'
    
    def test_weight_validation(self):
        """Test that weights sum to 1.0."""
        config = SearchConfig(
            vector_weight=0.4,
            keyword_weight=0.3,
            metadata_weight=0.3
        )
        
        total = config.vector_weight + config.keyword_weight + config.metadata_weight
        assert abs(total - 1.0) < 0.001, f"Weights should sum to 1.0, got {total}"


class TestRedisSearchIndex:
    """Test Redis-based keyword and metadata search."""
    
    @patch('src.core.retrieval.hybrid_search.redis.Redis')
    def test_redis_connection(self, mock_redis):
        """Test Redis connection initialization."""
        index = RedisSearchIndex('localhost', 6379)
        mock_redis.assert_called_once_with(host='localhost', port=6379, decode_responses=True)
    
    @patch('src.core.retrieval.hybrid_search.redis.Redis')
    def test_index_chunk(self, mock_redis):
        """Test indexing a chunk in Redis."""
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        
        index = RedisSearchIndex()
        
        chunk = Chunk(
            id="chunk_123",
            document_id="doc_456",
            content="Test content for indexing",
            position=0,
            metadata={'chunk_type': 'standard', 'tags': ['test', 'sample']}
        )
        
        document = Document(
            id="doc_456",
            title="Test Document",
            content="Full document content"
        )
        
        index.index_chunk(chunk, document)
        
        # Should call hset with correct data
        mock_client.hset.assert_called_once()
        call_args = mock_client.hset.call_args
        assert call_args[0][0] == "chunk:chunk_123"
        
        mapping = call_args[1]['mapping']
        assert mapping['content'] == "Test content for indexing"
        assert mapping['document_title'] == "Test Document"
        assert mapping['chunk_type'] == 'standard'
        assert mapping['tags'] == 'test,sample'
    
    @patch('src.core.retrieval.hybrid_search.redis.Redis')
    def test_search_keywords(self, mock_redis):
        """Test keyword search functionality."""
        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_client.ft.return_value = mock_search
        mock_redis.return_value = mock_client
        
        # Mock search results
        mock_doc = MagicMock()
        mock_doc.id = "chunk:123"
        mock_doc.score = 0.95
        mock_doc.content = "Test content"
        mock_doc.document_id = "doc_456"
        mock_doc.chunk_type = "standard"
        mock_doc.position = "0"
        mock_doc.tags = "test,sample"
        
        mock_results = MagicMock()
        mock_results.docs = [mock_doc]
        mock_search.search.return_value = mock_results
        
        index = RedisSearchIndex()
        results = index.search_keywords("test query", top_k=10)
        
        assert len(results) == 1
        assert results[0]['chunk_id'] == '123'
        assert results[0]['score'] == 0.95
        assert results[0]['content'] == "Test content"
    
    @patch('src.core.retrieval.hybrid_search.redis.Redis')
    def test_search_with_boost_terms(self, mock_redis):
        """Test keyword search with boost terms."""
        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_client.ft.return_value = mock_search
        mock_redis.return_value = mock_client
        
        mock_results = MagicMock()
        mock_results.docs = []
        mock_search.search.return_value = mock_results
        
        index = RedisSearchIndex()
        index.search_keywords(
            "important test query",
            top_k=10,
            boost_terms=['important', 'critical']
        )
        
        # Should boost 'important' term
        call_args = mock_search.search.call_args
        query_obj = call_args[0][0]
        # Query object would have boosted terms
    
    @patch('src.core.retrieval.hybrid_search.redis.Redis')
    def test_search_metadata(self, mock_redis):
        """Test metadata-based search."""
        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_client.ft.return_value = mock_search
        mock_redis.return_value = mock_client
        
        mock_doc = MagicMock()
        mock_doc.id = "chunk:789"
        mock_doc.content = "Metadata match content"
        mock_doc.document_id = "doc_111"
        mock_doc.chunk_type = "project"
        mock_doc.position = "5"
        mock_doc.tags = "architecture"
        
        mock_results = MagicMock()
        mock_results.docs = [mock_doc]
        mock_search.search.return_value = mock_results
        
        index = RedisSearchIndex()
        filters = {
            'document_id': 'doc_111',
            'chunk_type': 'project',
            'tags': ['architecture']
        }
        
        results = index.search_metadata(filters, top_k=5)
        
        assert len(results) == 1
        assert results[0]['chunk_id'] == '789'
        assert results[0]['score'] == 1.0  # Metadata matches are binary
    
    @patch('src.core.retrieval.hybrid_search.redis.Redis')
    def test_search_error_handling(self, mock_redis):
        """Test error handling in search operations."""
        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_client.ft.return_value = mock_search
        mock_redis.return_value = mock_client
        
        # Simulate search failure
        mock_search.search.side_effect = Exception("Redis search failed")
        
        index = RedisSearchIndex()
        results = index.search_keywords("test query")
        
        # Should return empty list on error
        assert results == []


class TestEnhancedHybridSearch:
    """Test the main EnhancedHybridSearch class."""
    
    @patch('src.core.retrieval.hybrid_search.VectorRetriever')
    @patch('src.core.retrieval.hybrid_search.RedisSearchIndex')
    def test_initialization(self, mock_redis_index, mock_vector_retriever):
        """Test EnhancedHybridSearch initialization."""
        search = EnhancedHybridSearch()
        
        assert search.config is not None
        assert search.vector_retriever is not None
        assert search.keyword_index is not None
        assert search.executor is not None
    
    @pytest.mark.asyncio
    @patch('src.core.retrieval.hybrid_search.VectorRetriever')
    @patch('src.core.retrieval.hybrid_search.RedisSearchIndex')
    async def test_parallel_search(self, mock_redis_index, mock_vector_retriever):
        """Test parallel execution of search strategies."""
        # Mock vector results
        vector_results = [
            RetrievalResult(
                chunk_id="chunk_1",
                content="Vector result 1",
                score=0.95,
                source="vector"
            ),
            RetrievalResult(
                chunk_id="chunk_2",
                content="Vector result 2",
                score=0.90,
                source="vector"
            )
        ]
        
        # Mock keyword results
        keyword_results = [
            {
                'chunk_id': 'chunk_2',
                'content': 'Keyword result 2',
                'score': 0.85,
                'document_id': 'doc_1',
                'metadata': {}
            },
            {
                'chunk_id': 'chunk_3',
                'content': 'Keyword result 3',
                'score': 0.80,
                'document_id': 'doc_2',
                'metadata': {}
            }
        ]
        
        mock_vector_retriever.return_value.retrieve.return_value = vector_results
        mock_redis_index.return_value.search_keywords.return_value = keyword_results
        mock_redis_index.return_value.search_metadata.return_value = []
        
        search = EnhancedHybridSearch()
        
        # Mock the async methods
        search._async_vector_search = AsyncMock(return_value=vector_results)
        search._async_keyword_search = AsyncMock(return_value=[
            RetrievalResult(chunk_id=r['chunk_id'], content=r['content'], 
                          score=r['score'], source='keyword')
            for r in keyword_results
        ])
        search._async_metadata_search = AsyncMock(return_value=[])
        
        config = SearchConfig(use_parallel=True)
        results = await search.search("test query", [0.1] * 384, config)
        
        # Should have called all search methods
        search._async_vector_search.assert_called_once()
        search._async_keyword_search.assert_called_once()
        
        # Results should be fused
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_reciprocal_rank_fusion(self):
        """Test RRF algorithm implementation."""
        search = EnhancedHybridSearch()
        
        # Create test result sets
        vector_results = [
            RetrievalResult(chunk_id="A", content="Content A", score=0.95, source="vector"),
            RetrievalResult(chunk_id="B", content="Content B", score=0.90, source="vector"),
            RetrievalResult(chunk_id="C", content="Content C", score=0.85, source="vector"),
        ]
        
        keyword_results = [
            RetrievalResult(chunk_id="B", content="Content B", score=0.88, source="keyword"),
            RetrievalResult(chunk_id="D", content="Content D", score=0.82, source="keyword"),
            RetrievalResult(chunk_id="A", content="Content A", score=0.78, source="keyword"),
        ]
        
        metadata_results = [
            RetrievalResult(chunk_id="C", content="Content C", score=1.0, source="metadata"),
            RetrievalResult(chunk_id="E", content="Content E", score=1.0, source="metadata"),
        ]
        
        result_sets = {
            'vector': vector_results,
            'keyword': keyword_results,
            'metadata': metadata_results
        }
        
        weights = [0.4, 0.3, 0.3]  # vector, keyword, metadata
        k = 60
        
        fused = search._reciprocal_rank_fusion(result_sets, weights, k)
        
        # Check RRF scoring
        assert len(fused) == 5  # A, B, C, D, E
        
        # Find chunk A's score (appears in vector rank 1, keyword rank 3)
        chunk_a = next(r for r in fused if r.chunk_id == "A")
        expected_a = weights[0] * (1/(k+1)) + weights[1] * (1/(k+3))
        assert abs(chunk_a.score - expected_a) < 0.001
        
        # Find chunk B's score (appears in vector rank 2, keyword rank 1)
        chunk_b = next(r for r in fused if r.chunk_id == "B")
        expected_b = weights[0] * (1/(k+2)) + weights[1] * (1/(k+1))
        assert abs(chunk_b.score - expected_b) < 0.001
        
        # Check that results are sorted by RRF score
        scores = [r.score for r in fused]
        assert scores == sorted(scores, reverse=True)
    
    def test_rrf_edge_cases(self):
        """Test RRF with edge cases."""
        search = EnhancedHybridSearch()
        
        # Empty result sets
        result_sets = {'vector': [], 'keyword': [], 'metadata': []}
        fused = search._reciprocal_rank_fusion(result_sets, [0.4, 0.3, 0.3], 60)
        assert fused == []
        
        # Single result
        result_sets = {
            'vector': [RetrievalResult(chunk_id="A", content="A", score=1.0, source="vector")],
            'keyword': [],
            'metadata': []
        }
        fused = search._reciprocal_rank_fusion(result_sets, [0.4, 0.3, 0.3], 60)
        assert len(fused) == 1
        assert fused[0].chunk_id == "A"
        
        # Duplicate handling
        result_sets = {
            'vector': [RetrievalResult(chunk_id="A", content="A", score=1.0, source="vector")],
            'keyword': [RetrievalResult(chunk_id="A", content="A", score=1.0, source="keyword")],
            'metadata': [RetrievalResult(chunk_id="A", content="A", score=1.0, source="metadata")]
        }
        fused = search._reciprocal_rank_fusion(result_sets, [0.4, 0.3, 0.3], 60)
        assert len(fused) == 1
        assert fused[0].source == "hybrid"  # Should be marked as hybrid
    
    @pytest.mark.asyncio
    async def test_search_with_config_override(self):
        """Test search with configuration override."""
        search = EnhancedHybridSearch()
        
        # Mock search methods
        search._async_vector_search = AsyncMock(return_value=[])
        search._async_keyword_search = AsyncMock(return_value=[])
        search._async_metadata_search = AsyncMock(return_value=[])
        
        # Custom config
        config = SearchConfig(
            vector_weight=0.6,
            keyword_weight=0.2,
            metadata_weight=0.2,
            vector_top_k=50,
            keyword_top_k=30,
            use_parallel=False
        )
        
        results = await search.search("test", [0.1] * 384, config)
        
        # Should use custom top_k values
        search._async_vector_search.assert_called_with([0.1] * 384, 50)
        search._async_keyword_search.assert_called_with("test", 30, None)
    
    def test_normalize_scores(self):
        """Test score normalization."""
        search = EnhancedHybridSearch()
        
        results = [
            RetrievalResult(chunk_id="A", content="A", score=10.0, source="test"),
            RetrievalResult(chunk_id="B", content="B", score=5.0, source="test"),
            RetrievalResult(chunk_id="C", content="C", score=2.0, source="test"),
        ]
        
        normalized = search._normalize_scores(results)
        
        # Scores should be in [0, 1] range
        assert normalized[0].score == 1.0  # Max score
        assert normalized[2].score == 0.0  # Min score
        assert 0 <= normalized[1].score <= 1.0
        
        # Test with identical scores
        same_results = [
            RetrievalResult(chunk_id="A", content="A", score=5.0, source="test"),
            RetrievalResult(chunk_id="B", content="B", score=5.0, source="test"),
        ]
        
        normalized = search._normalize_scores(same_results)
        assert all(r.score == 0.5 for r in normalized)
    
    def test_calculate_metrics(self):
        """Test retrieval metrics calculation."""
        search = EnhancedHybridSearch()
        
        retrieved = [
            RetrievalResult(chunk_id="A", content="A", score=0.9, source="test"),
            RetrievalResult(chunk_id="B", content="B", score=0.8, source="test"),
            RetrievalResult(chunk_id="C", content="C", score=0.7, source="test"),
            RetrievalResult(chunk_id="D", content="D", score=0.6, source="test"),
        ]
        
        ground_truth = ["A", "B", "E", "F"]
        
        metrics = search.calculate_metrics(retrieved, ground_truth)
        
        assert metrics['true_positives'] == 2  # A and B
        assert metrics['retrieved_count'] == 4
        assert metrics['relevant_count'] == 4
        assert metrics['precision'] == 0.5  # 2/4
        assert metrics['recall'] == 0.5  # 2/4
        assert metrics['f1'] == 0.5  # 2*0.5*0.5/(0.5+0.5)
        
        # Test with empty results
        metrics = search.calculate_metrics([], ground_truth)
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0


class TestAsyncMethods:
    """Test async wrapper methods."""
    
    @pytest.mark.asyncio
    @patch('src.core.retrieval.hybrid_search.VectorRetriever')
    async def test_async_vector_search(self, mock_vector_retriever):
        """Test async vector search wrapper."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            RetrievalResult(chunk_id="A", content="A", score=0.9, source="vector")
        ]
        mock_vector_retriever.return_value = mock_retriever
        
        search = EnhancedHybridSearch()
        results = await search._async_vector_search([0.1] * 384, 10)
        
        assert len(results) == 1
        assert results[0].chunk_id == "A"
        mock_retriever.retrieve.assert_called_once_with([0.1] * 384, 10)
    
    @pytest.mark.asyncio
    @patch('src.core.retrieval.hybrid_search.VectorRetriever')
    async def test_async_vector_search_error(self, mock_vector_retriever):
        """Test async vector search error handling."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = Exception("Vector search failed")
        mock_vector_retriever.return_value = mock_retriever
        
        search = EnhancedHybridSearch()
        results = await search._async_vector_search([0.1] * 384, 10)
        
        # Should return empty list on error
        assert results == []
    
    @pytest.mark.asyncio
    async def test_async_empty_results(self):
        """Test async empty results method."""
        search = EnhancedHybridSearch()
        results = await search._async_empty_results()
        assert results == []


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.mark.asyncio
    @patch('src.core.retrieval.hybrid_search.VectorRetriever')
    @patch('src.core.retrieval.hybrid_search.RedisSearchIndex')
    async def test_complete_search_flow(self, mock_redis, mock_vector):
        """Test complete search flow with all components."""
        # Setup mocks
        vector_results = [
            RetrievalResult(chunk_id=f"vec_{i}", content=f"Vector content {i}", 
                          score=0.9-i*0.1, source="vector")
            for i in range(5)
        ]
        
        keyword_results = [
            {'chunk_id': f'key_{i}', 'content': f'Keyword content {i}',
             'score': 0.85-i*0.1, 'document_id': f'doc_{i}', 'metadata': {}}
            for i in range(3)
        ]
        
        mock_vector.return_value.retrieve.return_value = vector_results
        mock_redis.return_value.search_keywords.return_value = keyword_results
        mock_redis.return_value.search_metadata.return_value = []
        
        search = EnhancedHybridSearch()
        
        # Override async methods
        search._async_vector_search = AsyncMock(return_value=vector_results)
        search._async_keyword_search = AsyncMock(return_value=[
            RetrievalResult(chunk_id=r['chunk_id'], content=r['content'],
                          score=r['score'], source='keyword')
            for r in keyword_results
        ])
        search._async_metadata_search = AsyncMock(return_value=[])
        
        # Perform search
        query = "test query for integration"
        embedding = np.random.rand(384).tolist()
        results = await search.search(query, embedding)
        
        # Verify results
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(0 <= r.score <= 1 for r in results)
        
        # Check that results are sorted by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_search_with_metadata_filters(self):
        """Test search with metadata filters."""
        search = EnhancedHybridSearch()
        
        # Mock methods
        search._async_vector_search = AsyncMock(return_value=[])
        search._async_keyword_search = AsyncMock(return_value=[])
        search._async_metadata_search = AsyncMock(return_value=[
            RetrievalResult(chunk_id="meta_1", content="Metadata match",
                          score=1.0, source="metadata")
        ])
        
        config = SearchConfig(
            metadata_filters={'doc_type': 'project', 'tags': ['important']}
        )
        
        results = await search.search("query", [0.1] * 384, config)
        
        # Should have called metadata search with filters
        search._async_metadata_search.assert_called_once()
        call_args = search._async_metadata_search.call_args
        assert call_args[0][0] == config.metadata_filters


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])