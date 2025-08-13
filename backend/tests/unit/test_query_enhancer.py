"""
Comprehensive tests for Query Enhancement with decomposition and expansion.
"""
import pytest
import asyncio
from unittest.mock import patch, Mock, AsyncMock
import json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.core.query.query_enhancer import (
    QueryEnhancer,
    QueryType,
    SubQuery,
    EnhancedQuery
)


class TestQueryTypeClassification:
    """Test query type detection and classification."""
    
    def test_factual_query_detection(self):
        """Test detection of factual queries."""
        enhancer = QueryEnhancer(bedrock_client=None, use_local_models=False)
        
        test_queries = [
            "What is user authentication?",
            "Who is the project manager?",
            "When did the project start?",
            "Where is the configuration file?",
            "Define REST API",
        ]
        
        for query in test_queries:
            query_type, _ = enhancer._analyze_query(query)
            assert query_type == QueryType.FACTUAL, f"'{query}' should be FACTUAL"
    
    def test_analytical_query_detection(self):
        """Test detection of analytical queries."""
        enhancer = QueryEnhancer(bedrock_client=None, use_local_models=False)
        
        test_queries = [
            "How does the authentication system work?",
            "Why does the system use Redis?",
            "Explain the architecture",
            "Analyze the performance bottlenecks",
            "What is the impact of caching?"
        ]
        
        for query in test_queries:
            query_type, _ = enhancer._analyze_query(query)
            assert query_type == QueryType.ANALYTICAL, f"'{query}' should be ANALYTICAL"
    
    def test_comparative_query_detection(self):
        """Test detection of comparative queries."""
        enhancer = QueryEnhancer(bedrock_client=None, use_local_models=False)
        
        test_queries = [
            "Compare Redis and Memcached",
            "What's the difference between REST and GraphQL?",
            "Which is better: PostgreSQL or MySQL?",
            "Redis versus DynamoDB performance",
            "JWT vs OAuth comparison"
        ]
        
        for query in test_queries:
            query_type, _ = enhancer._analyze_query(query)
            assert query_type == QueryType.COMPARATIVE, f"'{query}' should be COMPARATIVE"
    
    def test_navigational_query_detection(self):
        """Test detection of navigational queries."""
        enhancer = QueryEnhancer(bedrock_client=None, use_local_models=False)
        
        test_queries = [
            "Find the API documentation",
            "Locate the configuration file",
            "Show me the architecture diagram",
            "Where can I find the test results?",
            "Documentation for authentication"
        ]
        
        for query in test_queries:
            query_type, _ = enhancer._analyze_query(query)
            assert query_type == QueryType.NAVIGATIONAL, f"'{query}' should be NAVIGATIONAL"
    
    def test_multi_hop_query_detection(self):
        """Test detection of multi-hop queries."""
        enhancer = QueryEnhancer(bedrock_client=None, use_local_models=False)
        
        test_queries = [
            "How does authentication affect performance?",
            "What causes the database slowdown and how to fix it?",
            "How does caching lead to better response times?",
            "What results from improper configuration?",
            "How does A impact B and then affect C?"
        ]
        
        for query in test_queries:
            query_type, _ = enhancer._analyze_query(query)
            assert query_type == QueryType.MULTI_HOP, f"'{query}' should be MULTI_HOP"
    
    def test_exploratory_query_detection(self):
        """Test detection of exploratory queries."""
        enhancer = QueryEnhancer(bedrock_client=None, use_local_models=False)
        
        # Long, open-ended questions
        query = "What are all the different ways we can optimize the system performance and what trade-offs should we consider?"
        query_type, _ = enhancer._analyze_query(query)
        assert query_type == QueryType.EXPLORATORY
    
    def test_complexity_scoring(self):
        """Test query complexity scoring."""
        enhancer = QueryEnhancer(bedrock_client=None, use_local_models=False)
        
        # Simple query
        _, complexity = enhancer._analyze_query("What is Redis?")
        assert complexity < 0.3, "Simple query should have low complexity"
        
        # Medium complexity
        _, complexity = enhancer._analyze_query("How does Redis caching improve performance?")
        assert 0.3 <= complexity <= 0.7, "Medium query should have moderate complexity"
        
        # High complexity
        _, complexity = enhancer._analyze_query(
            "How does the authentication system interact with the caching layer, "
            "and what are the implications for performance and security?"
        )
        assert complexity > 0.5, "Complex query should have high complexity"


class TestDecompositionLogic:
    """Test query decomposition logic."""
    
    def test_needs_decomposition(self):
        """Test detection of queries needing decomposition."""
        enhancer = QueryEnhancer(bedrock_client=None, use_local_models=False)
        
        # Should need decomposition
        assert enhancer._needs_decomposition("How does A work and what is B?")
        assert enhancer._needs_decomposition("First do X then do Y")
        assert enhancer._needs_decomposition("What happened before and after the update?")
        assert enhancer._needs_decomposition("A, B, C, and D are important")
        assert enhancer._needs_decomposition("What is X? What is Y? What is Z?")
        assert enhancer._needs_decomposition("This is a very long query with many words that discusses multiple topics")
        
        # Should NOT need decomposition
        assert not enhancer._needs_decomposition("What is Redis?")
        assert not enhancer._needs_decomposition("Show me the docs")
        assert not enhancer._needs_decomposition("Simple query")
    
    def test_rule_based_decomposition(self):
        """Test rule-based query decomposition."""
        enhancer = QueryEnhancer(bedrock_client=None, use_local_models=False)
        
        # Test AND splitting
        query = "What is authentication and how does it work?"
        sub_queries = enhancer._rule_based_decompose(query, 5)
        
        assert len(sub_queries) == 2
        assert "authentication" in sub_queries[0].question
        assert "work" in sub_queries[1].question
        
        # Test comma splitting
        query = "Explain Redis, PostgreSQL, and Neo4j"
        sub_queries = enhancer._rule_based_decompose(query, 5)
        
        assert len(sub_queries) == 3
        assert any("Redis" in sq.question for sq in sub_queries)
        assert any("PostgreSQL" in sq.question for sq in sub_queries)
        assert any("Neo4j" in sq.question for sq in sub_queries)
        
        # Test question mark splitting
        query = "What is REST? How does it work? Why use it?"
        sub_queries = enhancer._rule_based_decompose(query, 5)
        
        assert len(sub_queries) == 3
        assert all(sq.question.endswith("?") for sq in sub_queries)
    
    def test_dependency_detection(self):
        """Test detection of dependencies between sub-queries."""
        enhancer = QueryEnhancer(bedrock_client=None, use_local_models=False)
        
        query = "What is authentication and how does it affect performance?"
        sub_queries = enhancer._rule_based_decompose(query, 5)
        
        # Second query should depend on first (uses "it")
        assert len(sub_queries) == 2
        if "it" in sub_queries[1].question.lower():
            assert 0 in sub_queries[1].dependencies
    
    def test_max_sub_queries_limit(self):
        """Test that decomposition respects max limit."""
        enhancer = QueryEnhancer(bedrock_client=None, use_local_models=False)
        
        query = "A and B and C and D and E and F and G"
        sub_queries = enhancer._rule_based_decompose(query, 3)
        
        assert len(sub_queries) <= 3
    
    @pytest.mark.asyncio
    @patch('src.core.query.query_enhancer.QueryEnhancer._llm_decompose')
    async def test_llm_decomposition_fallback(self, mock_llm):
        """Test fallback to rule-based when LLM fails."""
        enhancer = QueryEnhancer(bedrock_client=Mock(), use_local_models=False)
        
        # Simulate LLM failure
        mock_llm.side_effect = Exception("LLM failed")
        
        query = "What is A and what is B?"
        sub_queries = await enhancer._decompose_query(query, None, 5)
        
        # Should fall back to rule-based
        assert len(sub_queries) == 2
        assert "A" in sub_queries[0].question
        assert "B" in sub_queries[1].question


class TestEntityAndConceptExtraction:
    """Test entity and concept extraction."""
    
    @patch('spacy.load')
    def test_spacy_entity_extraction(self, mock_spacy):
        """Test entity extraction using spaCy."""
        # Mock spaCy components
        mock_doc = Mock()
        mock_ent1 = Mock(text="Redis")
        mock_ent2 = Mock(text="PostgreSQL")
        mock_doc.ents = [mock_ent1, mock_ent2]
        
        mock_token1 = Mock(pos_="NOUN", is_stop=False, text="database")
        mock_token2 = Mock(pos_="VERB", is_stop=False, text="connect")
        mock_token3 = Mock(pos_="DET", is_stop=True, text="the")
        mock_doc.__iter__ = Mock(return_value=iter([mock_token1, mock_token2, mock_token3]))
        
        mock_chunk = Mock(text="database system")
        mock_doc.noun_chunks = [mock_chunk]
        
        mock_nlp = Mock(return_value=mock_doc)
        mock_spacy.return_value = mock_nlp
        
        enhancer = QueryEnhancer(use_local_models=True)
        entities, concepts = enhancer._extract_entities_and_concepts(
            "Connect Redis to PostgreSQL database system"
        )
        
        assert "Redis" in entities
        assert "PostgreSQL" in entities
        assert "database" in concepts
        assert "database system" in concepts
    
    def test_fallback_entity_extraction(self):
        """Test fallback entity extraction without spaCy."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        query = "How does the authentication system work with Redis cache?"
        entities, concepts = enhancer._extract_entities_and_concepts(query)
        
        # Should extract non-stopword terms
        assert "authentication" in concepts
        assert "system" in concepts
        assert "redis" in concepts
        assert "cache" in concepts
        
        # Should not include stopwords
        assert "the" not in concepts
        assert "with" not in concepts
        assert "does" not in concepts
    
    def test_concept_deduplication(self):
        """Test that concepts are deduplicated while preserving order."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        query = "Redis Redis cache Redis database Redis"
        entities, concepts = enhancer._extract_entities_and_concepts(query)
        
        # Should have "redis" only once
        redis_count = concepts.count("redis")
        assert redis_count == 1
        
        # Should limit to top 10 concepts
        long_query = " ".join([f"concept{i}" for i in range(20)])
        entities, concepts = enhancer._extract_entities_and_concepts(long_query)
        assert len(concepts) <= 10


class TestTermExpansion:
    """Test term expansion with synonyms."""
    
    def test_predefined_synonym_expansion(self):
        """Test expansion using predefined synonyms."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        # Test auth expansion
        expanded = enhancer._expand_terms("user auth", ["auth"])
        assert "auth" in expanded
        assert "authentication" in expanded["auth"]
        assert "authorization" in expanded["auth"]
        assert "login" in expanded["auth"]
        
        # Test API expansion
        expanded = enhancer._expand_terms("REST API", ["api"])
        assert "api" in expanded
        assert "interface" in expanded["api"]
        assert "endpoint" in expanded["api"]
        
        # Test database expansion
        expanded = enhancer._expand_terms("db connection", ["db"])
        assert "db" in expanded
        assert "database" in expanded["db"]
        assert "datastore" in expanded["db"]
    
    def test_partial_match_expansion(self):
        """Test expansion with partial term matches."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        # "configuration" should match "config"
        expanded = enhancer._expand_terms("system configuration", ["configuration"])
        assert "configuration" in expanded
        assert "settings" in expanded["configuration"]
        assert "parameters" in expanded["configuration"]
    
    @patch('spacy.load')
    def test_morphological_variations(self, mock_spacy):
        """Test generation of morphological variations."""
        # Mock spaCy components
        mock_doc = Mock()
        mock_token = Mock(pos_="VERB", lemma_="create", text="creating")
        mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
        
        mock_nlp = Mock(return_value=mock_doc)
        mock_spacy.return_value = mock_nlp
        
        enhancer = QueryEnhancer(use_local_models=True)
        expanded = enhancer._expand_terms("creating users", [])
        
        assert "creating" in expanded
        expected_variations = ["create", "creates", "created"]
        for var in expected_variations:
            assert var in expanded["creating"]
    
    def test_no_expansion_for_unknown_terms(self):
        """Test that unknown terms don't get expanded."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        expanded = enhancer._expand_terms("xyz123 unknown", ["xyz123", "unknown"])
        
        # Unknown terms might not be in the result
        assert len(expanded) == 0 or all(
            term in enhancer.synonym_dict for term in expanded
        )


class TestQueryVariationGeneration:
    """Test generation of query variations."""
    
    def test_synonym_substitution_variations(self):
        """Test variations through synonym substitution."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        query = "user auth system"
        expanded_terms = {"auth": ["authentication", "login"]}
        
        variations = enhancer._generate_variations(query, expanded_terms)
        
        assert query in variations  # Original should be first
        assert any("authentication" in v for v in variations)
        assert any("login" in v for v in variations)
    
    def test_question_format_variations(self):
        """Test generation of different question formats."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        query = "How does caching work?"
        variations = enhancer._generate_variations(query, {})
        
        assert query in variations
        assert any("Can you explain" in v for v in variations)
        assert any("Tell me about" in v for v in variations)
        assert any("What do you know about" in v for v in variations)
    
    def test_variation_limit(self):
        """Test that variations are limited to prevent explosion."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        query = "test query"
        expanded_terms = {
            "test": ["exam", "check", "validate", "verify"],
            "query": ["question", "search", "request", "inquiry"]
        }
        
        variations = enhancer._generate_variations(query, expanded_terms)
        
        assert len(variations) <= 7  # Should be limited
        assert variations[0] == query  # Original first
    
    def test_no_duplicate_variations(self):
        """Test that variations are unique."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        query = "test test test"
        expanded_terms = {"test": ["check"]}
        
        variations = enhancer._generate_variations(query, expanded_terms)
        
        # Check for uniqueness (case-insensitive)
        lower_variations = [v.lower() for v in variations]
        assert len(lower_variations) == len(set(lower_variations))


class TestReasoningPathDetermination:
    """Test determination of reasoning paths for multi-hop queries."""
    
    def test_simple_reasoning_path(self):
        """Test reasoning path for simple dependencies."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        sub_queries = [
            SubQuery("What is A?", QueryType.FACTUAL, [], "high"),
            SubQuery("How does A work?", QueryType.ANALYTICAL, [0], "medium"),
            SubQuery("What is the impact?", QueryType.ANALYTICAL, [1], "low")
        ]
        
        path = enhancer._determine_reasoning_path(sub_queries)
        
        assert "Start with: What is A?" in path
        assert "Then use results from" in path
        assert "What is the impact?" in path
    
    def test_no_dependencies_path(self):
        """Test reasoning path with no dependencies."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        sub_queries = [
            SubQuery("What is A?", QueryType.FACTUAL, [], "high"),
            SubQuery("What is B?", QueryType.FACTUAL, [], "high"),
            SubQuery("What is C?", QueryType.FACTUAL, [], "high")
        ]
        
        path = enhancer._determine_reasoning_path(sub_queries)
        
        assert "Start with: What is A?, What is B?, What is C?" in path
    
    def test_complex_dependencies_path(self):
        """Test reasoning path with complex dependencies."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        sub_queries = [
            SubQuery("What is auth?", QueryType.FACTUAL, [], "high"),
            SubQuery("What is cache?", QueryType.FACTUAL, [], "high"),
            SubQuery("How do they interact?", QueryType.ANALYTICAL, [0, 1], "medium"),
            SubQuery("What is the performance impact?", QueryType.ANALYTICAL, [2], "low")
        ]
        
        path = enhancer._determine_reasoning_path(sub_queries)
        
        assert "Start with: What is auth?, What is cache?" in path
        assert "What is auth" in path
        assert "What is cache" in path
        assert "How do they interact?" in path
    
    def test_empty_sub_queries_path(self):
        """Test reasoning path with empty sub-queries."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        path = enhancer._determine_reasoning_path([])
        assert path is None


class TestResultCombination:
    """Test combining results from sub-queries."""
    
    def test_combine_sub_query_results(self):
        """Test combining results from multiple sub-queries."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        sub_queries = [
            SubQuery("What is Redis?", QueryType.FACTUAL, [], "high"),
            SubQuery("How does it work?", QueryType.ANALYTICAL, [0], "medium")
        ]
        
        sub_results = [
            ["Redis is an in-memory data store"],
            ["It works by storing data in RAM for fast access"]
        ]
        
        combined = enhancer.combine_sub_query_results(sub_queries, sub_results)
        
        assert "## What is Redis?" in combined
        assert "Redis is an in-memory data store" in combined
        assert "## How does it work?" in combined
        assert "storing data in RAM" in combined
    
    def test_combine_with_result_objects(self):
        """Test combining results that are objects with content attribute."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        sub_queries = [SubQuery("Test?", QueryType.FACTUAL, [], "high")]
        
        # Mock result objects
        mock_result = Mock()
        mock_result.content = "Test content"
        
        sub_results = [[mock_result]]
        
        combined = enhancer.combine_sub_query_results(sub_queries, sub_results)
        
        assert "Test content" in combined
    
    def test_combine_empty_results(self):
        """Test combining empty results."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        combined = enhancer.combine_sub_query_results([], [])
        assert combined == ""
        
        sub_queries = [SubQuery("Test?", QueryType.FACTUAL, [], "high")]
        combined = enhancer.combine_sub_query_results(sub_queries, [[]])
        assert "## Test?" in combined


class TestQueryEmbeddingWeights:
    """Test query embedding weight calculation."""
    
    def test_default_weights(self):
        """Test default embedding weights."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        enhanced_query = EnhancedQuery(
            original="test",
            sub_queries=[],
            expanded_terms={},
            variations=[],
            query_type=QueryType.FACTUAL,
            complexity_score=0.5,
            entities=[],
            key_concepts=[]
        )
        
        weights = enhancer.get_query_embedding_weights(enhanced_query)
        
        assert weights['original'] == 1.0
        assert weights['sub_queries'] == 0.8
        assert weights['variations'] == 0.6
        assert weights['expanded'] == 0.4
    
    def test_multi_hop_weights(self):
        """Test weights for multi-hop queries."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        enhanced_query = EnhancedQuery(
            original="test",
            sub_queries=[],
            expanded_terms={},
            variations=[],
            query_type=QueryType.MULTI_HOP,
            complexity_score=0.8,
            entities=[],
            key_concepts=[]
        )
        
        weights = enhancer.get_query_embedding_weights(enhanced_query)
        
        assert weights['sub_queries'] == 1.0  # Sub-queries more important
    
    def test_navigational_weights(self):
        """Test weights for navigational queries."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        enhanced_query = EnhancedQuery(
            original="test",
            sub_queries=[],
            expanded_terms={},
            variations=[],
            query_type=QueryType.NAVIGATIONAL,
            complexity_score=0.3,
            entities=[],
            key_concepts=[]
        )
        
        weights = enhancer.get_query_embedding_weights(enhanced_query)
        
        assert weights['expanded'] == 0.2  # Less emphasis on expansion


class TestIntegrationScenarios:
    """Test complete enhancement scenarios."""
    
    @pytest.mark.asyncio
    async def test_simple_query_enhancement(self):
        """Test enhancement of a simple query."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        query = "What is Redis?"
        enhanced = await enhancer.enhance_query(query)
        
        assert enhanced.original == query
        assert enhanced.query_type == QueryType.FACTUAL
        assert enhanced.complexity_score < 0.5
        assert len(enhanced.sub_queries) == 1
        assert enhanced.sub_queries[0].question == query
        assert "redis" in enhanced.key_concepts
    
    @pytest.mark.asyncio
    async def test_complex_query_enhancement(self):
        """Test enhancement of a complex query."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        query = "How does the authentication system work and what are the security implications?"
        enhanced = await enhancer.enhance_query(query)
        
        assert enhanced.original == query
        assert enhanced.complexity_score > 0.5
        assert len(enhanced.sub_queries) == 2
        assert "authentication" in str(enhanced.sub_queries[0].question).lower()
        assert "security" in str(enhanced.sub_queries[1].question).lower()
        assert len(enhanced.variations) > 1
    
    @pytest.mark.asyncio
    async def test_enhancement_with_context(self):
        """Test enhancement with additional context."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        query = "How does it work?"
        context = "We're discussing the Redis caching system"
        
        enhanced = await enhancer.enhance_query(query, context)
        
        assert enhanced.original == query
        # Context might influence decomposition if LLM was available
    
    @pytest.mark.asyncio
    async def test_max_sub_queries_respected(self):
        """Test that max sub-queries limit is respected."""
        enhancer = QueryEnhancer(use_local_models=False)
        
        query = "A and B and C and D and E and F and G and H"
        enhanced = await enhancer.enhance_query(query, max_sub_queries=3)
        
        assert len(enhanced.sub_queries) <= 3


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])