"""
Reranker Module - Advanced reranking strategies for retrieval results
"""
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from dataclasses import dataclass
import logging
from collections import defaultdict
import math

from ..models import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RerankConfig:
    """Configuration for reranking"""
    strategy: str = "cross_encoder"  # cross_encoder, bm25, semantic, hybrid
    top_k: int = 10
    diversity_weight: float = 0.3
    relevance_weight: float = 0.7
    use_mmr: bool = True
    mmr_lambda: float = 0.7
    

class Reranker:
    """Advanced reranking strategies for improving retrieval results"""
    
    def __init__(self, config: Optional[RerankConfig] = None):
        self.config = config or RerankConfig()
        self.rerank_cache = {}
        
    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        strategy: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Rerank results using specified strategy
        
        Args:
            query: Original query
            results: Initial retrieval results
            strategy: Reranking strategy to use
            
        Returns:
            Reranked results
        """
        if not results:
            return results
        
        strategy = strategy or self.config.strategy
        
        if strategy == "cross_encoder":
            return await self.cross_encoder_rerank(query, results)
        elif strategy == "bm25":
            return self.bm25_rerank(query, results)
        elif strategy == "semantic":
            return await self.semantic_rerank(query, results)
        elif strategy == "hybrid":
            return await self.hybrid_rerank(query, results)
        elif strategy == "learning_to_rank":
            return await self.learning_to_rank(query, results)
        else:
            logger.warning(f"Unknown reranking strategy: {strategy}")
            return results
    
    async def cross_encoder_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    ) -> List[RetrievalResult]:
        """
        Rerank using cross-encoder model
        
        Args:
            query: Query text
            results: Results to rerank
            model_name: Cross-encoder model to use
            
        Returns:
            Reranked results
        """
        try:
            # In production, would use actual cross-encoder model
            # For now, simulate with enhanced scoring
            reranked = []
            
            for result in results:
                # Simulate cross-encoder scoring
                query_terms = set(query.lower().split())
                result_terms = set(result.content.lower().split())
                
                # Calculate overlap and position-based score
                overlap = len(query_terms & result_terms)
                coverage = overlap / len(query_terms) if query_terms else 0
                
                # Find query terms positions in result
                position_score = 0
                result_words = result.content.lower().split()
                for term in query_terms:
                    if term in result_words:
                        # Earlier positions get higher scores
                        pos = result_words.index(term)
                        position_score += 1.0 / (1 + pos * 0.1)
                
                # Combine scores
                cross_encoder_score = (
                    coverage * 0.4 +
                    position_score * 0.3 +
                    result.score * 0.3
                )
                
                reranked_result = RetrievalResult(
                    id=result.id,
                    content=result.content,
                    score=cross_encoder_score,
                    metadata={
                        **result.metadata,
                        'rerank_strategy': 'cross_encoder',
                        'original_score': result.score,
                        'coverage': coverage,
                        'position_score': position_score
                    }
                )
                reranked.append(reranked_result)
            
            # Sort by new scores
            reranked.sort(key=lambda x: x.score, reverse=True)
            
            # Apply MMR if configured
            if self.config.use_mmr:
                reranked = self.apply_mmr_reranking(
                    reranked,
                    lambda_param=self.config.mmr_lambda
                )
            
            return reranked[:self.config.top_k]
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {e}")
            return results[:self.config.top_k]
    
    def bm25_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        k1: float = 1.2,
        b: float = 0.75
    ) -> List[RetrievalResult]:
        """
        Rerank using BM25 scoring
        
        Args:
            query: Query text
            results: Results to rerank
            k1: BM25 k1 parameter
            b: BM25 b parameter
            
        Returns:
            BM25 reranked results
        """
        # Tokenize
        query_terms = query.lower().split()
        
        # Calculate document statistics
        doc_lengths = [len(r.content.split()) for r in results]
        avg_doc_length = np.mean(doc_lengths) if doc_lengths else 1
        
        # Calculate IDF scores
        doc_freq = defaultdict(int)
        for result in results:
            doc_terms = set(result.content.lower().split())
            for term in query_terms:
                if term in doc_terms:
                    doc_freq[term] += 1
        
        N = len(results)
        idf = {}
        for term in query_terms:
            df = doc_freq.get(term, 0)
            idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
        
        # Calculate BM25 scores
        reranked = []
        for i, result in enumerate(results):
            doc_terms = result.content.lower().split()
            doc_length = doc_lengths[i]
            
            score = 0
            for term in query_terms:
                tf = doc_terms.count(term)
                if tf > 0:
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * doc_length / avg_doc_length)
                    score += idf[term] * (numerator / denominator)
            
            # Combine with original score
            combined_score = score * 0.6 + result.score * 0.4
            
            reranked_result = RetrievalResult(
                id=result.id,
                content=result.content,
                score=combined_score,
                metadata={
                    **result.metadata,
                    'rerank_strategy': 'bm25',
                    'bm25_score': score,
                    'original_score': result.score
                }
            )
            reranked.append(reranked_result)
        
        # Sort by BM25 scores
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        return reranked[:self.config.top_k]
    
    async def semantic_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        similarity_threshold: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Rerank based on semantic similarity
        
        Args:
            query: Query text
            results: Results to rerank
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Semantically reranked results
        """
        # In production, would use sentence transformers
        # For now, use enhanced keyword matching
        
        query_concepts = self._extract_concepts(query)
        reranked = []
        
        for result in results:
            result_concepts = self._extract_concepts(result.content)
            
            # Calculate concept overlap
            concept_similarity = self._calculate_concept_similarity(
                query_concepts,
                result_concepts
            )
            
            # Calculate semantic score
            semantic_score = (
                concept_similarity * 0.5 +
                result.score * 0.5
            )
            
            if semantic_score >= similarity_threshold:
                reranked_result = RetrievalResult(
                    id=result.id,
                    content=result.content,
                    score=semantic_score,
                    metadata={
                        **result.metadata,
                        'rerank_strategy': 'semantic',
                        'concept_similarity': concept_similarity,
                        'original_score': result.score
                    }
                )
                reranked.append(reranked_result)
        
        # Sort by semantic scores
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        # Apply diversity if configured
        if self.config.diversity_weight > 0:
            reranked = self.apply_diversity_reranking(
                reranked,
                diversity_weight=self.config.diversity_weight
            )
        
        return reranked[:self.config.top_k]
    
    async def hybrid_rerank(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Hybrid reranking combining multiple strategies
        
        Args:
            query: Query text
            results: Results to rerank
            
        Returns:
            Hybrid reranked results
        """
        # Get scores from different strategies
        bm25_results = self.bm25_rerank(query, results)
        semantic_results = await self.semantic_rerank(query, results)
        
        # Create score dictionaries
        bm25_scores = {r.id: r.score for r in bm25_results}
        semantic_scores = {r.id: r.score for r in semantic_results}
        original_scores = {r.id: r.score for r in results}
        
        # Combine scores
        hybrid_results = []
        for result in results:
            doc_id = result.id
            
            # Weighted combination
            hybrid_score = (
                bm25_scores.get(doc_id, 0) * 0.3 +
                semantic_scores.get(doc_id, 0) * 0.4 +
                original_scores.get(doc_id, 0) * 0.3
            )
            
            hybrid_result = RetrievalResult(
                id=doc_id,
                content=result.content,
                score=hybrid_score,
                metadata={
                    **result.metadata,
                    'rerank_strategy': 'hybrid',
                    'bm25_score': bm25_scores.get(doc_id, 0),
                    'semantic_score': semantic_scores.get(doc_id, 0),
                    'original_score': original_scores.get(doc_id, 0)
                }
            )
            hybrid_results.append(hybrid_result)
        
        # Sort by hybrid scores
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        
        return hybrid_results[:self.config.top_k]
    
    async def learning_to_rank(
        self,
        query: str,
        results: List[RetrievalResult],
        features: Optional[Dict[str, Callable]] = None
    ) -> List[RetrievalResult]:
        """
        Learning to rank approach using feature engineering
        
        Args:
            query: Query text
            results: Results to rerank
            features: Feature extraction functions
            
        Returns:
            LTR reranked results
        """
        # Default features if not provided
        if not features:
            features = {
                'query_coverage': lambda q, r: self._calculate_query_coverage(q, r),
                'term_proximity': lambda q, r: self._calculate_term_proximity(q, r),
                'length_norm': lambda q, r: self._calculate_length_normalization(r),
                'freshness': lambda q, r: self._calculate_freshness(r),
                'popularity': lambda q, r: self._calculate_popularity(r)
            }
        
        # Extract features for each result
        feature_vectors = []
        for result in results:
            feature_vector = []
            for feature_name, feature_func in features.items():
                try:
                    feature_value = feature_func(query, result)
                    feature_vector.append(feature_value)
                except Exception as e:
                    logger.error(f"Error extracting feature {feature_name}: {e}")
                    feature_vector.append(0)
            
            feature_vectors.append(feature_vector)
        
        # Normalize features
        feature_vectors = np.array(feature_vectors)
        if len(feature_vectors) > 0:
            # Min-max normalization
            mins = feature_vectors.min(axis=0)
            maxs = feature_vectors.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1  # Avoid division by zero
            feature_vectors = (feature_vectors - mins) / ranges
        
        # Simple linear combination (in production, would use trained model)
        weights = np.array([0.3, 0.2, 0.1, 0.2, 0.2])  # Example weights
        if len(weights) != feature_vectors.shape[1]:
            weights = np.ones(feature_vectors.shape[1]) / feature_vectors.shape[1]
        
        ltr_scores = feature_vectors @ weights
        
        # Create reranked results
        reranked = []
        for i, result in enumerate(results):
            ltr_result = RetrievalResult(
                id=result.id,
                content=result.content,
                score=float(ltr_scores[i]),
                metadata={
                    **result.metadata,
                    'rerank_strategy': 'learning_to_rank',
                    'features': dict(zip(features.keys(), feature_vectors[i])),
                    'original_score': result.score
                }
            )
            reranked.append(ltr_result)
        
        # Sort by LTR scores
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        return reranked[:self.config.top_k]
    
    def apply_mmr_reranking(
        self,
        results: List[RetrievalResult],
        lambda_param: float = 0.7
    ) -> List[RetrievalResult]:
        """
        Apply Maximal Marginal Relevance for diversity
        
        Args:
            results: Results to diversify
            lambda_param: Trade-off parameter
            
        Returns:
            Diversified results
        """
        if len(results) <= 1:
            return results
        
        selected = [results[0]]
        remaining = results[1:]
        
        while remaining and len(selected) < self.config.top_k:
            best_score = -1
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                # Relevance score
                relevance = candidate.score
                
                # Calculate diversity
                min_similarity = 1.0
                for selected_result in selected:
                    similarity = self._calculate_similarity(
                        candidate.content,
                        selected_result.content
                    )
                    min_similarity = min(min_similarity, similarity)
                
                # MMR score
                mmr_score = (
                    lambda_param * relevance +
                    (1 - lambda_param) * (1 - min_similarity)
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(remaining[best_idx])
                remaining.pop(best_idx)
            else:
                break
        
        return selected
    
    def apply_diversity_reranking(
        self,
        results: List[RetrievalResult],
        diversity_weight: float = 0.3
    ) -> List[RetrievalResult]:
        """
        Apply diversity-based reranking
        
        Args:
            results: Results to diversify
            diversity_weight: Weight for diversity
            
        Returns:
            Diversified results
        """
        if len(results) <= 1:
            return results
        
        # Calculate pairwise similarities
        n = len(results)
        similarities = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._calculate_similarity(
                    results[i].content,
                    results[j].content
                )
                similarities[i, j] = sim
                similarities[j, i] = sim
        
        # Rerank with diversity
        reranked = []
        used_indices = set()
        
        # Start with highest scoring
        best_idx = 0
        reranked.append(results[best_idx])
        used_indices.add(best_idx)
        
        # Iteratively add diverse results
        while len(reranked) < min(self.config.top_k, len(results)):
            best_score = -1
            best_idx = -1
            
            for i in range(n):
                if i in used_indices:
                    continue
                
                # Calculate average similarity to selected
                avg_similarity = np.mean([
                    similarities[i, j] for j in used_indices
                ])
                
                # Combined score
                diversity_score = 1 - avg_similarity
                combined_score = (
                    (1 - diversity_weight) * results[i].score +
                    diversity_weight * diversity_score
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = i
            
            if best_idx >= 0:
                reranked.append(results[best_idx])
                used_indices.add(best_idx)
            else:
                break
        
        return reranked
    
    # Helper methods
    def _extract_concepts(self, text: str) -> Set[str]:
        """Extract key concepts from text"""
        # Simple noun phrase extraction
        words = text.lower().split()
        concepts = set()
        
        # Single words
        for word in words:
            if len(word) > 3:  # Skip short words
                concepts.add(word)
        
        # Bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            concepts.add(bigram)
        
        return concepts
    
    def _calculate_concept_similarity(
        self,
        concepts1: Set[str],
        concepts2: Set[str]
    ) -> float:
        """Calculate similarity between concept sets"""
        if not concepts1 or not concepts2:
            return 0.0
        
        intersection = len(concepts1 & concepts2)
        union = len(concepts1 | concepts2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_query_coverage(self, query: str, result: RetrievalResult) -> float:
        """Calculate how well result covers query terms"""
        query_terms = set(query.lower().split())
        result_terms = set(result.content.lower().split())
        
        if not query_terms:
            return 0.0
        
        covered = len(query_terms & result_terms)
        return covered / len(query_terms)
    
    def _calculate_term_proximity(self, query: str, result: RetrievalResult) -> float:
        """Calculate proximity of query terms in result"""
        query_terms = query.lower().split()
        result_words = result.content.lower().split()
        
        if not query_terms or not result_words:
            return 0.0
        
        positions = []
        for term in query_terms:
            if term in result_words:
                positions.append(result_words.index(term))
        
        if len(positions) < 2:
            return 0.0
        
        # Calculate average distance between terms
        positions.sort()
        distances = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        avg_distance = np.mean(distances)
        
        # Normalize (closer is better)
        return 1.0 / (1 + avg_distance)
    
    def _calculate_length_normalization(self, result: RetrievalResult) -> float:
        """Calculate length normalization score"""
        length = len(result.content.split())
        
        # Prefer medium-length documents
        optimal_length = 200
        deviation = abs(length - optimal_length)
        
        return 1.0 / (1 + deviation / optimal_length)
    
    def _calculate_freshness(self, result: RetrievalResult) -> float:
        """Calculate freshness score based on metadata"""
        # Check if result has timestamp metadata
        if 'timestamp' in result.metadata:
            # In production, would calculate based on actual timestamp
            return 0.8  # Placeholder
        return 0.5  # Default freshness
    
    def _calculate_popularity(self, result: RetrievalResult) -> float:
        """Calculate popularity score based on metadata"""
        # Check if result has popularity indicators
        if 'views' in result.metadata:
            views = result.metadata['views']
            return min(1.0, views / 1000)  # Normalize
        return 0.5  # Default popularity
    
    def clear_cache(self):
        """Clear the rerank cache"""
        self.rerank_cache.clear()