"""
Integration tests for the Fusion Controller with all retrieval strategies.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import numpy as np
import time
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.core.retrieval.fusion_controller import (
    FusionController,
    FusionConfig,
    RetrievalStrategy
)
from src.core.retrieval.hybrid_search import SearchConfig
from src.core.query.query_enhancer import QueryEnhancer, EnhancedQuery, QueryType
from src.models import RetrievalResult, Chunk, Document


class TestFusionControllerIntegration:
    """Test FusionController with all components integrated."""
    
    @pytest.fixture
    def fusion_controller(self):
        """Create a FusionController instance for testing."""
        config = FusionConfig(
            enable_temporal=True,
            enable_graph_rag=True,
            enable_query_enhancement=True,
            temporal_weight=0.3,
            rerank_top_k=10
        )
        return FusionController(config)
    
    @pytest.mark.asyncio
    async def test_complete_retrieval_pipeline(self, fusion_controller):
        """Test the complete retrieval pipeline with all strategies."""
        # Mock all components
        with patch.object(fusion_controller, 'hybrid_search') as mock_hybrid, \
             patch.object(fusion_controller, 'graph_rag') as mock_graph, \
             patch.object(fusion_controller, 'query_enhancer') as mock_enhancer:
            
            # Setup mock enhanced query
            mock_enhanced = EnhancedQuery(
                original="How does caching improve performance?",
                sub_queries=[
                    Mock(question="What is caching?", type=QueryType.FACTUAL),
                    Mock(question="How does caching work?", type=QueryType.ANALYTICAL)
                ],
                expanded_terms={"caching": ["cache", "Redis", "Memcached"]},
                variations=["How does cache improve speed?"],
                query_type=QueryType.ANALYTICAL,
                complexity_score=0.7,
                entities=["caching", "performance"],
                key_concepts=["caching", "performance", "optimization"]
            )
            mock_enhancer.enhance_query = AsyncMock(return_value=mock_enhanced)
            
            # Setup mock search results
            mock_hybrid.search = AsyncMock(return_value=[
                RetrievalResult(
                    chunk_id="chunk_1",
                    content="Caching stores frequently accessed data in memory",
                    score=0.9,
                    source="hybrid",
                    metadata={"created_at_ms": time.time() * 1000}
                ),
                RetrievalResult(
                    chunk_id="chunk_2",
                    content="Redis is an in-memory cache",
                    score=0.85,
                    source="hybrid",
                    metadata={"created_at_ms": time.time() * 1000 - 86400000}
                )
            ])
            
            # Setup mock graph results
            mock_graph.multi_hop_reasoning = AsyncMock(return_value=[
                RetrievalResult(
                    chunk_id="graph_1",
                    content="Caching reduces database load which improves response time",
                    score=0.88,
                    source="graph",
                    metadata={"reasoning_path": "cache->db_load->response_time"}
                )
            ])
            
            # Execute retrieval
            query = "How does caching improve performance?"
            embedding = np.random.rand(384).tolist()
            results = await fusion_controller.retrieve(query, embedding)
            
            # Verify results
            assert len(results) > 0
            assert all(isinstance(r, RetrievalResult) for r in results)
            
            # Check that all strategies were used
            mock_enhancer.enhance_query.assert_called_once()
            mock_hybrid.search.assert_called()
            mock_graph.multi_hop_reasoning.assert_called()
    
    @pytest.mark.asyncio
    async def test_temporal_scoring_integration(self, fusion_controller):
        """Test temporal scoring integration with retrieval."""
        fusion_controller.config.enable_temporal = True
        
        # Create results with different ages
        now_ms = time.time() * 1000
        results = [
            RetrievalResult(
                chunk_id="new",
                content="Latest documentation",
                score=0.8,
                metadata={
                    "created_at_ms": now_ms,
                    "doc_type": "project"
                }
            ),
            RetrievalResult(
                chunk_id="old",
                content="Old documentation",
                score=0.9,  # Higher semantic score
                metadata={
                    "created_at_ms": now_ms - 180 * 86400000,  # 180 days old
                    "doc_type": "project"
                }
            ),
            RetrievalResult(
                chunk_id="values",
                content="Company values",
                score=0.7,
                metadata={
                    "created_at_ms": now_ms - 365 * 86400000,  # 1 year old
                    "doc_type": "values"
                }
            )
        ]
        
        # Apply temporal scoring with recency query
        scored = fusion_controller._apply_temporal_scoring(results, "latest project docs")
        
        # New doc should rank higher despite lower semantic score
        assert scored[0].chunk_id == "new"
        # Values doc should maintain score (no decay)
        values_result = next(r for r in scored if r.chunk_id == "values")
        assert values_result.score >= 0.7
    
    @pytest.mark.asyncio
    async def test_query_enhancement_flow(self, fusion_controller):
        """Test query enhancement and decomposition flow."""
        with patch.object(fusion_controller, 'query_enhancer') as mock_enhancer:
            # Complex query that needs decomposition
            query = "How does the authentication system work and what are the security implications?"
            
            mock_enhanced = EnhancedQuery(
                original=query,
                sub_queries=[
                    Mock(question="How does the authentication system work?", 
                         type=QueryType.ANALYTICAL, dependencies=[]),
                    Mock(question="What are the security implications?",
                         type=QueryType.ANALYTICAL, dependencies=[0])
                ],
                expanded_terms={
                    "authentication": ["auth", "login", "OAuth", "JWT"],
                    "security": ["secure", "vulnerability", "protection"]
                },
                variations=[],
                query_type=QueryType.MULTI_HOP,
                complexity_score=0.8,
                entities=["authentication", "security"],
                key_concepts=["authentication", "security", "system"]
            )
            
            mock_enhancer.enhance_query = AsyncMock(return_value=mock_enhanced)
            
            # Process query
            enhanced = await fusion_controller._enhance_query(query)
            
            assert enhanced.query_type == QueryType.MULTI_HOP
            assert len(enhanced.sub_queries) == 2
            assert enhanced.complexity_score == 0.8
            assert "auth" in enhanced.expanded_terms["authentication"]
    
    @pytest.mark.asyncio
    async def test_hybrid_search_with_rrf(self, fusion_controller):
        """Test hybrid search with Reciprocal Rank Fusion."""
        with patch.object(fusion_controller, 'hybrid_search') as mock_hybrid:
            # Setup different result sources
            vector_results = [
                RetrievalResult(chunk_id="v1", content="Vector 1", score=0.95, source="vector"),
                RetrievalResult(chunk_id="v2", content="Vector 2", score=0.90, source="vector")
            ]
            
            keyword_results = [
                RetrievalResult(chunk_id="k1", content="Keyword 1", score=0.88, source="keyword"),
                RetrievalResult(chunk_id="v1", content="Vector 1", score=0.85, source="keyword")  # Overlap
            ]
            
            # Mock will return combined results
            mock_hybrid.search = AsyncMock(return_value=vector_results + keyword_results)
            
            # Execute search
            results = await fusion_controller._hybrid_search(
                "test query",
                [0.1] * 384,
                SearchConfig(use_parallel=True)
            )
            
            # Check RRF fusion occurred (overlapping docs should be boosted)
            assert len(results) == 4
            mock_hybrid.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_graph_rag_multi_hop(self, fusion_controller):
        """Test GraphRAG multi-hop reasoning integration."""
        with patch.object(fusion_controller, 'graph_rag') as mock_graph:
            # Setup multi-hop reasoning chain
            mock_graph.multi_hop_reasoning = AsyncMock(return_value=[
                RetrievalResult(
                    chunk_id="hop1",
                    content="A causes B",
                    score=0.9,
                    source="graph",
                    metadata={"hop": 1, "relation": "causes"}
                ),
                RetrievalResult(
                    chunk_id="hop2",
                    content="B leads to C",
                    score=0.85,
                    source="graph",
                    metadata={"hop": 2, "relation": "leads_to"}
                ),
                RetrievalResult(
                    chunk_id="hop3",
                    content="C affects performance",
                    score=0.8,
                    source="graph",
                    metadata={"hop": 3, "relation": "affects"}
                )
            ])
            
            # Execute multi-hop query
            results = await fusion_controller._graph_multi_hop(
                "How does A affect performance?",
                max_hops=3
            )
            
            assert len(results) == 3
            assert results[0].metadata["hop"] == 1
            assert results[-1].content == "C affects performance"
    
    @pytest.mark.asyncio
    async def test_result_deduplication(self, fusion_controller):
        """Test deduplication of results from multiple sources."""
        # Results with duplicates
        results = [
            RetrievalResult(chunk_id="1", content="Content A", score=0.9, source="vector"),
            RetrievalResult(chunk_id="1", content="Content A", score=0.85, source="keyword"),
            RetrievalResult(chunk_id="2", content="Content B", score=0.8, source="vector"),
            RetrievalResult(chunk_id="1", content="Content A", score=0.88, source="graph")
        ]
        
        deduped = fusion_controller._deduplicate_results(results)
        
        # Should keep highest scoring version of duplicates
        assert len(deduped) == 2
        chunk1 = next(r for r in deduped if r.chunk_id == "1")
        assert chunk1.score == 0.9  # Highest score for chunk 1
        assert chunk1.source == "hybrid"  # Marked as hybrid when from multiple sources
    
    @pytest.mark.asyncio
    async def test_reranking_pipeline(self, fusion_controller):
        """Test the reranking pipeline."""
        # Initial results
        results = [
            RetrievalResult(chunk_id="1", content="Relevant content", score=0.7),
            RetrievalResult(chunk_id="2", content="Very relevant", score=0.8),
            RetrievalResult(chunk_id="3", content="Somewhat relevant", score=0.6),
            RetrievalResult(chunk_id="4", content="Highly relevant", score=0.9),
            RetrievalResult(chunk_id="5", content="Not relevant", score=0.4)
        ]
        
        with patch.object(fusion_controller, 'reranker') as mock_reranker:
            mock_reranker.rerank = AsyncMock(return_value=[
                results[3],  # Highly relevant
                results[1],  # Very relevant
                results[0],  # Relevant
                results[2],  # Somewhat relevant
            ])
            
            fusion_controller.config.rerank_top_k = 3
            reranked = await fusion_controller._rerank_results(results, "test query")
            
            assert len(reranked) == 3  # Top-k filtering
            assert reranked[0].chunk_id == "4"  # Best result first


class TestFusionStrategies:
    """Test different fusion strategies."""
    
    @pytest.mark.asyncio
    async def test_strategy_selection(self):
        """Test automatic strategy selection based on query type."""
        controller = FusionController()
        
        # Factual query - should use basic hybrid search
        strategy = controller._select_strategy("What is Redis?")
        assert strategy == RetrievalStrategy.HYBRID
        
        # Multi-hop query - should use graph RAG
        strategy = controller._select_strategy("How does A cause B which affects C?")
        assert strategy == RetrievalStrategy.GRAPH_MULTI_HOP
        
        # Recent query - should emphasize temporal
        strategy = controller._select_strategy("What are the latest updates?")
        assert strategy == RetrievalStrategy.TEMPORAL_HYBRID
        
        # Complex analytical - should use enhanced
        strategy = controller._select_strategy(
            "Analyze the system architecture and explain the trade-offs"
        )
        assert strategy == RetrievalStrategy.ENHANCED_ANALYTICAL
    
    @pytest.mark.asyncio
    async def test_weighted_fusion(self):
        """Test weighted fusion of different retrieval strategies."""
        controller = FusionController()
        
        # Results from different strategies with weights
        strategy_results = {
            "hybrid": ([
                RetrievalResult(chunk_id="h1", content="Hybrid 1", score=0.9),
                RetrievalResult(chunk_id="h2", content="Hybrid 2", score=0.8)
            ], 0.4),
            "graph": ([
                RetrievalResult(chunk_id="g1", content="Graph 1", score=0.85),
                RetrievalResult(chunk_id="h1", content="Hybrid 1", score=0.82)  # Overlap
            ], 0.3),
            "temporal": ([
                RetrievalResult(chunk_id="t1", content="Temporal 1", score=0.88),
                RetrievalResult(chunk_id="h2", content="Hybrid 2", score=0.75)  # Overlap
            ], 0.3)
        }
        
        fused = controller._weighted_fusion(strategy_results)
        
        # Check weighted scores
        assert len(fused) == 4  # h1, h2, g1, t1
        
        # h1 should have combined score from hybrid and graph
        h1_result = next(r for r in fused if r.chunk_id == "h1")
        expected_h1 = 0.4 * 0.9 + 0.3 * 0.82  # Weighted combination
        assert abs(h1_result.score - expected_h1) < 0.01
    
    @pytest.mark.asyncio
    async def test_adaptive_strategy_switching(self):
        """Test adaptive switching between strategies based on results."""
        controller = FusionController()
        controller.config.adaptive_strategy = True
        
        with patch.object(controller, 'hybrid_search') as mock_hybrid, \
             patch.object(controller, 'graph_rag') as mock_graph:
            
            # First attempt with hybrid returns poor results
            mock_hybrid.search = AsyncMock(return_value=[
                RetrievalResult(chunk_id="1", content="Weak match", score=0.4)
            ])
            
            # Fallback to graph RAG returns better results
            mock_graph.multi_hop_reasoning = AsyncMock(return_value=[
                RetrievalResult(chunk_id="2", content="Good match", score=0.85)
            ])
            
            results = await controller.retrieve_adaptive(
                "Complex query",
                [0.1] * 384,
                min_score_threshold=0.6
            )
            
            # Should have used graph RAG due to poor hybrid results
            assert len(results) == 1
            assert results[0].chunk_id == "2"
            mock_graph.multi_hop_reasoning.assert_called()


class TestTemporalIntegration:
    """Test temporal features integration."""
    
    @pytest.mark.asyncio
    async def test_temporal_query_routing(self):
        """Test routing queries based on temporal intent."""
        controller = FusionController()
        
        # Historical query
        historical_results = await controller._route_temporal_query(
            "Find historical meeting notes from Q1",
            time_range=(
                datetime(2024, 1, 1).timestamp() * 1000,
                datetime(2024, 3, 31).timestamp() * 1000
            )
        )
        
        # Recent query
        recent_results = await controller._route_temporal_query(
            "Latest project updates",
            time_range=(
                (datetime.now() - timedelta(days=7)).timestamp() * 1000,
                datetime.now().timestamp() * 1000
            )
        )
        
        # Different strategies should be used
        assert historical_results != recent_results
    
    @pytest.mark.asyncio
    async def test_temporal_decay_curves(self):
        """Test application of temporal decay curves."""
        controller = FusionController()
        
        now_ms = time.time() * 1000
        
        # Test different document types
        test_cases = [
            ("project", 120, 0.37),  # Project at characteristic life
            ("meeting", 14, 0.37),   # Meeting at characteristic life
            ("values", 365, 1.0),    # Values never decay
            ("default", 60, 0.37)    # Default at characteristic life
        ]
        
        for doc_type, age_days, expected_score in test_cases:
            result = RetrievalResult(
                chunk_id="test",
                content="Test",
                score=1.0,
                metadata={
                    "created_at_ms": now_ms - age_days * 86400000,
                    "doc_type": doc_type
                }
            )
            
            scored = controller._apply_temporal_decay([result])[0]
            
            if doc_type == "values":
                assert scored.score == 1.0
            else:
                assert abs(scored.temporal_score - expected_score) < 0.05
    
    @pytest.mark.asyncio
    async def test_temporal_metadata_extraction(self):
        """Test extraction and use of temporal metadata."""
        controller = FusionController()
        
        with patch('src.core.temporal.date_extractor.extract_temporal_metadata') as mock_extract:
            mock_extract.return_value = {
                'created_at_ms': 1700000000000,
                'expires_at_ms': 1800000000000,
                'sprint': 23,
                'milestone': 'Q1 Release',
                'lifecycle_stage': 'active'
            }
            
            result = RetrievalResult(
                chunk_id="test",
                content="Sprint 23 objectives for Q1 Release",
                score=0.8,
                metadata={}
            )
            
            enriched = controller._enrich_temporal_metadata([result])[0]
            
            assert enriched.metadata['sprint'] == 23
            assert enriched.metadata['milestone'] == 'Q1 Release'
            assert enriched.metadata['lifecycle_stage'] == 'active'


class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""
    
    @pytest.mark.asyncio
    async def test_component_failure_resilience(self):
        """Test resilience when individual components fail."""
        controller = FusionController()
        
        with patch.object(controller, 'hybrid_search') as mock_hybrid, \
             patch.object(controller, 'graph_rag') as mock_graph, \
             patch.object(controller, 'query_enhancer') as mock_enhancer:
            
            # Graph RAG fails
            mock_graph.multi_hop_reasoning = AsyncMock(
                side_effect=Exception("Graph database unavailable")
            )
            
            # Other components work
            mock_hybrid.search = AsyncMock(return_value=[
                RetrievalResult(chunk_id="1", content="Result", score=0.8)
            ])
            
            mock_enhancer.enhance_query = AsyncMock(return_value=Mock(
                original="test",
                sub_queries=[],
                query_type=QueryType.FACTUAL
            ))
            
            # Should still return results from working components
            results = await controller.retrieve("test query", [0.1] * 384)
            
            assert len(results) > 0
            assert results[0].chunk_id == "1"
    
    @pytest.mark.asyncio
    async def test_empty_results_handling(self):
        """Test handling of empty results from all strategies."""
        controller = FusionController()
        
        with patch.object(controller, 'hybrid_search') as mock_hybrid, \
             patch.object(controller, 'graph_rag') as mock_graph:
            
            mock_hybrid.search = AsyncMock(return_value=[])
            mock_graph.multi_hop_reasoning = AsyncMock(return_value=[])
            
            results = await controller.retrieve("impossible query", [0.1] * 384)
            
            assert results == []
    
    @pytest.mark.asyncio
    async def test_invalid_embedding_handling(self):
        """Test handling of invalid embeddings."""
        controller = FusionController()
        
        # Invalid embedding dimensions
        with pytest.raises(ValueError):
            await controller.retrieve("test", [0.1] * 100)  # Wrong size
        
        # Non-numeric embedding
        with pytest.raises(TypeError):
            await controller.retrieve("test", ["not", "numeric"])
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling for long-running operations."""
        controller = FusionController()
        controller.config.timeout_seconds = 1
        
        with patch.object(controller, 'hybrid_search') as mock_hybrid:
            async def slow_search(*args):
                await asyncio.sleep(5)  # Longer than timeout
                return []
            
            mock_hybrid.search = slow_search
            
            # Should timeout and return partial results
            results = await controller.retrieve_with_timeout("test", [0.1] * 384)
            
            # Should have handled timeout gracefully
            assert isinstance(results, list)


class TestPerformanceOptimization:
    """Test performance optimizations in integration."""
    
    @pytest.mark.asyncio
    async def test_parallel_strategy_execution(self):
        """Test parallel execution of retrieval strategies."""
        controller = FusionController()
        controller.config.parallel_strategies = True
        
        call_times = []
        
        async def mock_search(*args):
            start = time.time()
            await asyncio.sleep(0.1)  # Simulate work
            call_times.append(time.time() - start)
            return [RetrievalResult(chunk_id=f"r{len(call_times)}", content="", score=0.8)]
        
        with patch.object(controller, 'hybrid_search') as mock_hybrid, \
             patch.object(controller, 'graph_rag') as mock_graph:
            
            mock_hybrid.search = mock_search
            mock_graph.multi_hop_reasoning = mock_search
            
            start = time.time()
            results = await controller.retrieve_parallel("test", [0.1] * 384)
            total_time = time.time() - start
            
            # Should execute in parallel (faster than sequential)
            assert total_time < 0.15  # Less than sequential (0.2)
            assert len(results) >= 2
    
    @pytest.mark.asyncio
    async def test_result_caching(self):
        """Test caching of retrieval results."""
        controller = FusionController()
        controller.config.enable_cache = True
        controller.cache = {}
        
        call_count = 0
        
        async def mock_search(*args):
            nonlocal call_count
            call_count += 1
            return [RetrievalResult(chunk_id="cached", content="Result", score=0.9)]
        
        with patch.object(controller, 'hybrid_search') as mock_hybrid:
            mock_hybrid.search = mock_search
            
            # First call
            results1 = await controller.retrieve_cached("test query", [0.1] * 384)
            assert call_count == 1
            
            # Second call (should use cache)
            results2 = await controller.retrieve_cached("test query", [0.1] * 384)
            assert call_count == 1  # No additional calls
            
            assert results1 == results2
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing of multiple queries."""
        controller = FusionController()
        
        queries = [
            ("Query 1", [0.1] * 384),
            ("Query 2", [0.2] * 384),
            ("Query 3", [0.3] * 384)
        ]
        
        with patch.object(controller, 'retrieve') as mock_retrieve:
            mock_retrieve.return_value = [
                RetrievalResult(chunk_id="1", content="Result", score=0.8)
            ]
            
            results = await controller.batch_retrieve(queries)
            
            assert len(results) == 3
            assert mock_retrieve.call_count == 3


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])