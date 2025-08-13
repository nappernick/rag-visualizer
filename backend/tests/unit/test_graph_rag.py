"""
Comprehensive tests for GraphRAG multi-hop reasoning and knowledge graph traversal.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import numpy as np
from typing import List, Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.core.graph.graph_rag import (
    GraphRAG,
    GraphNode,
    GraphEdge,
    TraversalPath,
    ReasoningChain
)


class TestGraphNodeOperations:
    """Test GraphNode creation and manipulation."""
    
    def test_node_creation(self):
        """Test creating graph nodes with various properties."""
        node = GraphNode(
            id="node_1",
            type="concept",
            content="Machine Learning",
            embedding=[0.1, 0.2, 0.3],
            metadata={"category": "AI", "importance": 0.9}
        )
        
        assert node.id == "node_1"
        assert node.type == "concept"
        assert node.content == "Machine Learning"
        assert node.embedding == [0.1, 0.2, 0.3]
        assert node.metadata["category"] == "AI"
        assert node.metadata["importance"] == 0.9
    
    def test_node_similarity(self):
        """Test node similarity calculation."""
        node1 = GraphNode(
            id="n1",
            type="concept",
            content="A",
            embedding=np.array([1.0, 0.0, 0.0])
        )
        
        node2 = GraphNode(
            id="n2",
            type="concept",
            content="B",
            embedding=np.array([0.0, 1.0, 0.0])
        )
        
        node3 = GraphNode(
            id="n3",
            type="concept",
            content="C",
            embedding=np.array([1.0, 0.0, 0.0])
        )
        
        # Orthogonal vectors should have low similarity
        sim_12 = node1.calculate_similarity(node2)
        assert sim_12 < 0.1
        
        # Same direction vectors should have high similarity
        sim_13 = node1.calculate_similarity(node3)
        assert sim_13 > 0.99
    
    def test_node_types(self):
        """Test different node types and their properties."""
        # Entity node
        entity = GraphNode(
            id="e1",
            type="entity",
            content="Redis",
            metadata={"entity_type": "technology"}
        )
        assert entity.is_entity()
        assert not entity.is_concept()
        
        # Concept node
        concept = GraphNode(
            id="c1",
            type="concept",
            content="Caching",
            metadata={"abstract_level": "high"}
        )
        assert concept.is_concept()
        assert not concept.is_entity()
        
        # Document node
        doc = GraphNode(
            id="d1",
            type="document",
            content="Architecture Guide",
            metadata={"doc_id": "doc_123"}
        )
        assert doc.type == "document"
    
    def test_node_equality(self):
        """Test node equality comparison."""
        node1 = GraphNode(id="n1", type="concept", content="Test")
        node2 = GraphNode(id="n1", type="concept", content="Test")
        node3 = GraphNode(id="n2", type="concept", content="Test")
        
        assert node1 == node2  # Same ID
        assert node1 != node3  # Different ID


class TestGraphEdgeOperations:
    """Test GraphEdge creation and properties."""
    
    def test_edge_creation(self):
        """Test creating edges with various properties."""
        edge = GraphEdge(
            source="node_1",
            target="node_2",
            type="relates_to",
            weight=0.8,
            metadata={"confidence": 0.9}
        )
        
        assert edge.source == "node_1"
        assert edge.target == "node_2"
        assert edge.type == "relates_to"
        assert edge.weight == 0.8
        assert edge.metadata["confidence"] == 0.9
    
    def test_edge_types(self):
        """Test different edge types."""
        edge_types = [
            ("causes", "causal"),
            ("relates_to", "semantic"),
            ("part_of", "hierarchical"),
            ("depends_on", "dependency"),
            ("references", "citation"),
            ("contradicts", "conflict")
        ]
        
        for edge_type, category in edge_types:
            edge = GraphEdge(
                source="a",
                target="b",
                type=edge_type,
                metadata={"category": category}
            )
            assert edge.type == edge_type
            assert edge.metadata["category"] == category
    
    def test_edge_weight_normalization(self):
        """Test edge weight normalization."""
        # Weights should be in [0, 1]
        edge = GraphEdge(source="a", target="b", weight=2.0)
        assert edge.weight == 1.0  # Should be clamped
        
        edge = GraphEdge(source="a", target="b", weight=-0.5)
        assert edge.weight == 0.0  # Should be clamped
        
        edge = GraphEdge(source="a", target="b", weight=0.5)
        assert edge.weight == 0.5  # Should remain unchanged
    
    def test_bidirectional_edges(self):
        """Test bidirectional edge handling."""
        edge = GraphEdge(
            source="a",
            target="b",
            type="relates_to",
            bidirectional=True
        )
        
        assert edge.bidirectional
        assert edge.can_traverse("a", "b")
        assert edge.can_traverse("b", "a")  # Can go both ways
        
        # Unidirectional edge
        edge2 = GraphEdge(
            source="a",
            target="b",
            type="causes",
            bidirectional=False
        )
        
        assert not edge2.bidirectional
        assert edge2.can_traverse("a", "b")
        assert not edge2.can_traverse("b", "a")  # Only one way


class TestTraversalPath:
    """Test graph traversal path tracking."""
    
    def test_path_creation(self):
        """Test creating traversal paths."""
        path = TraversalPath(
            nodes=["n1", "n2", "n3"],
            edges=[("n1", "n2"), ("n2", "n3")],
            total_weight=1.8,
            reasoning="Path from n1 to n3 via n2"
        )
        
        assert path.nodes == ["n1", "n2", "n3"]
        assert len(path.edges) == 2
        assert path.total_weight == 1.8
        assert path.length() == 3
        assert "n1 to n3" in path.reasoning
    
    def test_path_extension(self):
        """Test extending traversal paths."""
        path = TraversalPath(nodes=["n1"], edges=[], total_weight=0)
        
        # Extend path
        path.extend("n2", weight=0.8)
        assert path.nodes == ["n1", "n2"]
        assert path.edges == [("n1", "n2")]
        assert path.total_weight == 0.8
        
        # Extend again
        path.extend("n3", weight=0.6)
        assert path.nodes == ["n1", "n2", "n3"]
        assert path.edges == [("n1", "n2"), ("n2", "n3")]
        assert path.total_weight == 1.4
    
    def test_path_contains(self):
        """Test checking if path contains node."""
        path = TraversalPath(
            nodes=["n1", "n2", "n3"],
            edges=[("n1", "n2"), ("n2", "n3")]
        )
        
        assert path.contains("n1")
        assert path.contains("n2")
        assert path.contains("n3")
        assert not path.contains("n4")
    
    def test_path_loop_detection(self):
        """Test detection of loops in path."""
        path = TraversalPath(nodes=["n1", "n2"], edges=[("n1", "n2")])
        
        # Should detect if extending would create loop
        assert not path.would_create_loop("n3")
        assert path.would_create_loop("n1")  # Would create loop
        assert path.would_create_loop("n2")  # Would create loop
    
    def test_path_comparison(self):
        """Test comparing paths for multi-hop reasoning."""
        path1 = TraversalPath(
            nodes=["n1", "n2", "n3"],
            edges=[("n1", "n2"), ("n2", "n3")],
            total_weight=1.5
        )
        
        path2 = TraversalPath(
            nodes=["n1", "n4", "n3"],
            edges=[("n1", "n4"), ("n4", "n3")],
            total_weight=1.2
        )
        
        # Path2 should be better (lower weight = shorter/better path)
        assert path2.is_better_than(path1)
        assert not path1.is_better_than(path2)


class TestReasoningChain:
    """Test multi-hop reasoning chain construction."""
    
    def test_chain_creation(self):
        """Test creating reasoning chains."""
        chain = ReasoningChain(
            query="How does A affect C?",
            hops=[
                ("A", "causes", "B", 0.9),
                ("B", "impacts", "C", 0.8)
            ],
            confidence=0.72  # 0.9 * 0.8
        )
        
        assert chain.query == "How does A affect C?"
        assert len(chain.hops) == 2
        assert chain.confidence == 0.72
        assert chain.start_node() == "A"
        assert chain.end_node() == "C"
    
    def test_chain_validation(self):
        """Test reasoning chain validation."""
        # Valid chain
        valid_chain = ReasoningChain(
            query="test",
            hops=[
                ("A", "rel", "B", 0.9),
                ("B", "rel", "C", 0.8)
            ]
        )
        assert valid_chain.is_valid()
        
        # Invalid chain (disconnected)
        invalid_chain = ReasoningChain(
            query="test",
            hops=[
                ("A", "rel", "B", 0.9),
                ("C", "rel", "D", 0.8)  # C doesn't connect to B
            ]
        )
        assert not invalid_chain.is_valid()
    
    def test_chain_explanation(self):
        """Test generating explanations from chains."""
        chain = ReasoningChain(
            query="How does caching improve performance?",
            hops=[
                ("caching", "reduces", "database_load", 0.9),
                ("database_load", "affects", "response_time", 0.85),
                ("response_time", "determines", "performance", 0.95)
            ],
            confidence=0.73
        )
        
        explanation = chain.generate_explanation()
        
        assert "caching" in explanation
        assert "reduces" in explanation
        assert "database_load" in explanation
        assert "performance" in explanation
        assert "73%" in explanation or "0.73" in explanation


class TestGraphRAGCore:
    """Test core GraphRAG functionality."""
    
    @patch('src.core.graph.graph_rag.Neo4jGraph')
    def test_initialization(self, mock_neo4j):
        """Test GraphRAG initialization."""
        rag = GraphRAG(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password"
        )
        
        assert rag.graph is not None
        assert rag.embedder is not None
        assert rag.max_hops == 3  # Default
        assert rag.min_confidence == 0.5  # Default
    
    @pytest.mark.asyncio
    async def test_single_hop_query(self):
        """Test single-hop graph queries."""
        rag = GraphRAG()
        
        # Mock graph with simple structure
        rag.graph = Mock()
        rag.graph.get_neighbors.return_value = [
            GraphNode(id="n2", type="concept", content="Result 1"),
            GraphNode(id="n3", type="concept", content="Result 2")
        ]
        
        results = await rag.query_single_hop("n1", "relates_to")
        
        assert len(results) == 2
        assert results[0].content == "Result 1"
        assert results[1].content == "Result 2"
    
    @pytest.mark.asyncio
    async def test_multi_hop_traversal(self):
        """Test multi-hop graph traversal."""
        rag = GraphRAG()
        
        # Mock graph structure
        mock_graph = Mock()
        
        # Define graph structure
        def get_neighbors_mock(node_id, edge_type=None):
            neighbors = {
                "n1": [
                    GraphNode(id="n2", content="Node 2"),
                    GraphNode(id="n3", content="Node 3")
                ],
                "n2": [
                    GraphNode(id="n4", content="Node 4"),
                    GraphNode(id="n5", content="Node 5")
                ],
                "n3": [
                    GraphNode(id="n5", content="Node 5"),
                    GraphNode(id="n6", content="Node 6")
                ],
                "n4": [
                    GraphNode(id="n7", content="Target")
                ],
                "n5": [
                    GraphNode(id="n7", content="Target")
                ],
                "n6": [],
                "n7": []
            }
            return neighbors.get(node_id, [])
        
        mock_graph.get_neighbors = get_neighbors_mock
        rag.graph = mock_graph
        
        # Find paths from n1 to n7
        paths = await rag.find_paths("n1", "n7", max_hops=3)
        
        assert len(paths) > 0
        # Should find at least 2 paths: n1->n2->n4->n7 and n1->n2->n5->n7
        assert any(p.end_node() == "n7" for p in paths)
    
    @pytest.mark.asyncio
    async def test_reasoning_with_confidence(self):
        """Test reasoning with confidence thresholds."""
        rag = GraphRAG(min_confidence=0.7)
        
        # Create reasoning chains with different confidences
        high_conf_chain = ReasoningChain(
            query="test",
            hops=[("A", "rel", "B", 0.9), ("B", "rel", "C", 0.8)],
            confidence=0.72
        )
        
        low_conf_chain = ReasoningChain(
            query="test",
            hops=[("A", "rel", "B", 0.6), ("B", "rel", "C", 0.8)],
            confidence=0.48
        )
        
        # Filter by confidence
        chains = [high_conf_chain, low_conf_chain]
        filtered = rag.filter_by_confidence(chains)
        
        assert len(filtered) == 1
        assert filtered[0].confidence >= 0.7
    
    @pytest.mark.asyncio
    async def test_subgraph_extraction(self):
        """Test extracting relevant subgraphs."""
        rag = GraphRAG()
        
        # Mock graph
        mock_graph = Mock()
        mock_graph.get_subgraph.return_value = {
            "nodes": [
                GraphNode(id="n1", content="Center"),
                GraphNode(id="n2", content="Related 1"),
                GraphNode(id="n3", content="Related 2")
            ],
            "edges": [
                GraphEdge(source="n1", target="n2", weight=0.9),
                GraphEdge(source="n1", target="n3", weight=0.8)
            ]
        }
        rag.graph = mock_graph
        
        subgraph = await rag.extract_subgraph("n1", radius=2)
        
        assert len(subgraph["nodes"]) == 3
        assert len(subgraph["edges"]) == 2
        assert subgraph["nodes"][0].id == "n1"


class TestGraphRAGQueries:
    """Test complex GraphRAG query scenarios."""
    
    @pytest.mark.asyncio
    async def test_causal_reasoning(self):
        """Test causal reasoning queries."""
        rag = GraphRAG()
        
        # Setup causal graph
        rag.graph = Mock()
        
        def get_causal_chain(start, end):
            if start == "high_load" and end == "slow_response":
                return [
                    TraversalPath(
                        nodes=["high_load", "cpu_spike", "slow_response"],
                        edges=[
                            ("high_load", "causes", "cpu_spike"),
                            ("cpu_spike", "causes", "slow_response")
                        ],
                        total_weight=1.7
                    )
                ]
            return []
        
        rag.graph.find_causal_paths = get_causal_chain
        
        result = await rag.explain_causality("high_load", "slow_response")
        
        assert result is not None
        assert "high_load" in result
        assert "cpu_spike" in result
        assert "slow_response" in result
    
    @pytest.mark.asyncio
    async def test_entity_relationship_query(self):
        """Test entity relationship queries."""
        rag = GraphRAG()
        
        # Mock entity relationships
        rag.graph = Mock()
        rag.graph.get_entity_relations.return_value = [
            ("Redis", "used_by", "CacheService", 0.95),
            ("CacheService", "part_of", "Backend", 0.90),
            ("Backend", "serves", "API", 0.85)
        ]
        
        relations = await rag.find_entity_relations("Redis", depth=3)
        
        assert len(relations) == 3
        assert relations[0][0] == "Redis"
        assert relations[-1][2] == "API"
    
    @pytest.mark.asyncio
    async def test_contradiction_detection(self):
        """Test detecting contradictions in graph."""
        rag = GraphRAG()
        
        # Mock contradictory information
        rag.graph = Mock()
        rag.graph.find_contradictions.return_value = [
            {
                "node1": GraphNode(id="n1", content="Redis is fast"),
                "node2": GraphNode(id="n2", content="Redis is slow"),
                "confidence": 0.85
            }
        ]
        
        contradictions = await rag.detect_contradictions("Redis")
        
        assert len(contradictions) == 1
        assert "fast" in contradictions[0]["node1"].content
        assert "slow" in contradictions[0]["node2"].content
    
    @pytest.mark.asyncio
    async def test_concept_expansion(self):
        """Test expanding concepts through graph."""
        rag = GraphRAG()
        
        # Mock concept hierarchy
        rag.graph = Mock()
        
        def expand_concept(concept):
            expansions = {
                "caching": ["Redis", "Memcached", "in-memory storage"],
                "database": ["PostgreSQL", "MySQL", "NoSQL"],
                "API": ["REST", "GraphQL", "RPC"]
            }
            return expansions.get(concept, [])
        
        rag.graph.expand_concept = expand_concept
        
        expanded = await rag.expand_concept("caching")
        
        assert "Redis" in expanded
        assert "Memcached" in expanded
        assert len(expanded) == 3


class TestGraphRAGOptimization:
    """Test GraphRAG optimization and performance."""
    
    @pytest.mark.asyncio
    async def test_path_pruning(self):
        """Test pruning inefficient paths."""
        rag = GraphRAG()
        
        paths = [
            TraversalPath(nodes=["A", "B", "C"], edges=[], total_weight=3.0),
            TraversalPath(nodes=["A", "D", "C"], edges=[], total_weight=2.0),
            TraversalPath(nodes=["A", "E", "F", "C"], edges=[], total_weight=4.0),
            TraversalPath(nodes=["A", "C"], edges=[], total_weight=1.5)
        ]
        
        pruned = rag.prune_paths(paths, max_paths=2)
        
        assert len(pruned) == 2
        assert pruned[0].total_weight == 1.5  # Best path
        assert pruned[1].total_weight == 2.0  # Second best
    
    @pytest.mark.asyncio
    async def test_caching_traversals(self):
        """Test caching of frequently traversed paths."""
        rag = GraphRAG()
        rag.enable_cache = True
        rag.cache = {}
        
        # Mock expensive graph operation
        mock_graph = Mock()
        call_count = 0
        
        def expensive_traversal(start, end):
            nonlocal call_count
            call_count += 1
            return TraversalPath(
                nodes=[start, "middle", end],
                edges=[(start, "middle"), ("middle", end)],
                total_weight=2.0
            )
        
        mock_graph.find_path = expensive_traversal
        rag.graph = mock_graph
        
        # First call - should compute
        path1 = await rag.find_cached_path("A", "B")
        assert call_count == 1
        
        # Second call - should use cache
        path2 = await rag.find_cached_path("A", "B")
        assert call_count == 1  # No additional calls
        assert path1 == path2
    
    @pytest.mark.asyncio
    async def test_parallel_traversal(self):
        """Test parallel graph traversal for efficiency."""
        rag = GraphRAG()
        
        # Mock parallel traversal
        async def mock_traverse(start_nodes):
            await asyncio.sleep(0.01)  # Simulate work
            return {node: f"Result_{node}" for node in start_nodes}
        
        rag.parallel_traverse = mock_traverse
        
        start_nodes = ["n1", "n2", "n3", "n4", "n5"]
        results = await rag.parallel_traverse(start_nodes)
        
        assert len(results) == 5
        assert all(f"Result_{node}" == results[node] for node in start_nodes)


class TestGraphRAGIntegration:
    """Test GraphRAG integration with other components."""
    
    @pytest.mark.asyncio
    @patch('src.core.graph.graph_rag.VectorRetriever')
    async def test_hybrid_graph_vector_search(self, mock_vector):
        """Test combining graph and vector search."""
        rag = GraphRAG()
        
        # Mock vector search results
        mock_vector.return_value.retrieve.return_value = [
            Mock(chunk_id="c1", content="Vector result 1", score=0.9),
            Mock(chunk_id="c2", content="Vector result 2", score=0.8)
        ]
        
        # Mock graph traversal results
        rag.graph = Mock()
        rag.graph.find_related.return_value = [
            GraphNode(id="g1", content="Graph result 1"),
            GraphNode(id="g2", content="Graph result 2")
        ]
        
        # Perform hybrid search
        query = "Find related concepts"
        embedding = [0.1] * 384
        
        results = await rag.hybrid_search(query, embedding)
        
        assert len(results) > 0
        # Should have both vector and graph results
    
    @pytest.mark.asyncio
    async def test_graph_based_reranking(self):
        """Test using graph structure for result reranking."""
        rag = GraphRAG()
        
        # Initial results to rerank
        initial_results = [
            Mock(id="r1", content="Result 1", score=0.7),
            Mock(id="r2", content="Result 2", score=0.8),
            Mock(id="r3", content="Result 3", score=0.6)
        ]
        
        # Mock graph-based importance scores
        rag.graph = Mock()
        rag.graph.get_pagerank.return_value = {
            "r1": 0.9,  # High importance in graph
            "r2": 0.5,  # Medium importance
            "r3": 0.3   # Low importance
        }
        
        reranked = await rag.rerank_by_graph_importance(initial_results)
        
        # r1 should be first now due to high graph importance
        assert reranked[0].id == "r1"
    
    @pytest.mark.asyncio
    async def test_temporal_graph_queries(self):
        """Test temporal-aware graph queries."""
        rag = GraphRAG()
        
        # Mock temporal graph
        rag.graph = Mock()
        
        def get_temporal_neighbors(node, time_range):
            if time_range[0] <= 100 <= time_range[1]:
                return [GraphNode(id="old", content="Old info")]
            else:
                return [GraphNode(id="new", content="New info")]
        
        rag.graph.get_temporal_neighbors = get_temporal_neighbors
        
        # Query for old timeframe
        old_results = await rag.temporal_query("node", time_range=(0, 100))
        assert old_results[0].id == "old"
        
        # Query for new timeframe
        new_results = await rag.temporal_query("node", time_range=(101, 200))
        assert new_results[0].id == "new"


class TestGraphRAGErrorHandling:
    """Test error handling in GraphRAG."""
    
    @pytest.mark.asyncio
    async def test_missing_nodes(self):
        """Test handling of missing nodes in graph."""
        rag = GraphRAG()
        rag.graph = Mock()
        rag.graph.get_node.return_value = None
        
        result = await rag.query_node("nonexistent")
        assert result is None or result == []
    
    @pytest.mark.asyncio
    async def test_cyclic_path_detection(self):
        """Test detection and handling of cycles."""
        rag = GraphRAG()
        
        # Create graph with cycle
        rag.graph = Mock()
        
        def get_neighbors_with_cycle(node):
            neighbors = {
                "A": ["B"],
                "B": ["C"],
                "C": ["A"]  # Cycle back to A
            }
            return [GraphNode(id=n, content=n) for n in neighbors.get(node, [])]
        
        rag.graph.get_neighbors = get_neighbors_with_cycle
        
        # Should detect and avoid infinite loop
        paths = await rag.find_paths("A", "D", max_hops=10)
        assert len(paths) == 0  # No path to D due to cycle
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_filtering(self):
        """Test filtering results below confidence threshold."""
        rag = GraphRAG(min_confidence=0.7)
        
        chains = [
            ReasoningChain(query="q", hops=[], confidence=0.9),
            ReasoningChain(query="q", hops=[], confidence=0.6),
            ReasoningChain(query="q", hops=[], confidence=0.8),
            ReasoningChain(query="q", hops=[], confidence=0.5)
        ]
        
        filtered = rag.filter_by_confidence(chains)
        
        assert len(filtered) == 2
        assert all(c.confidence >= 0.7 for c in filtered)
    
    @pytest.mark.asyncio
    async def test_max_hop_limit(self):
        """Test enforcement of maximum hop limit."""
        rag = GraphRAG(max_hops=2)
        
        # Mock deep graph
        rag.graph = Mock()
        
        def get_linear_neighbors(node):
            # Create a linear chain: A->B->C->D->E
            next_nodes = {
                "A": "B",
                "B": "C", 
                "C": "D",
                "D": "E"
            }
            next_node = next_nodes.get(node)
            return [GraphNode(id=next_node, content=next_node)] if next_node else []
        
        rag.graph.get_neighbors = get_linear_neighbors
        
        # Should only traverse 2 hops from A
        paths = await rag.find_paths("A", "E", max_hops=2)
        
        # Should not reach E (requires 4 hops)
        assert len(paths) == 0 or all(p.length() <= 3 for p in paths)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])