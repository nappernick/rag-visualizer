"""
Merge Strategies - Different algorithms for merging retrieval results
"""
from typing import List, Dict, Any
import numpy as np
from collections import defaultdict

from ..models import RetrievalResult


class MergeStrategies:
    """Collection of strategies for merging retrieval results"""
    
    @staticmethod
    def weighted_merge(
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        vector_weight: float = 0.7,
        graph_weight: float = 0.3
    ) -> List[RetrievalResult]:
        """
        Merge results using weighted combination
        
        Args:
            vector_results: Results from vector search
            graph_results: Results from graph search
            vector_weight: Weight for vector results
            graph_weight: Weight for graph results
            
        Returns:
            Merged and weighted results
        """
        # Create score dictionary
        scores = {}
        content_map = {}
        
        # Add vector results with weight
        for result in vector_results:
            doc_id = result.id
            scores[doc_id] = scores.get(doc_id, 0) + (result.score * vector_weight)
            content_map[doc_id] = result
        
        # Add graph results with weight
        for result in graph_results:
            doc_id = result.id
            scores[doc_id] = scores.get(doc_id, 0) + (result.score * graph_weight)
            if doc_id not in content_map:
                content_map[doc_id] = result
        
        # Create merged results
        merged_results = []
        for doc_id, combined_score in scores.items():
            result = content_map[doc_id]
            merged_result = RetrievalResult(
                id=doc_id,
                content=result.content,
                score=combined_score,
                metadata={
                    **result.metadata,
                    'merge_strategy': 'weighted',
                    'original_score': result.score
                }
            )
            merged_results.append(merged_result)
        
        # Sort by combined score
        merged_results.sort(key=lambda x: x.score, reverse=True)
        
        return merged_results
    
    @staticmethod
    def reciprocal_rank_fusion(
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        k: int = 60
    ) -> List[RetrievalResult]:
        """
        Merge using Reciprocal Rank Fusion (RRF)
        
        Args:
            vector_results: Results from vector search
            graph_results: Results from graph search
            k: RRF constant (default 60)
            
        Returns:
            Merged results using RRF
        """
        rrf_scores = defaultdict(float)
        content_map = {}
        
        # Calculate RRF scores for vector results
        for rank, result in enumerate(vector_results):
            doc_id = result.id
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            content_map[doc_id] = result
        
        # Calculate RRF scores for graph results
        for rank, result in enumerate(graph_results):
            doc_id = result.id
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            if doc_id not in content_map:
                content_map[doc_id] = result
        
        # Create merged results
        merged_results = []
        for doc_id, rrf_score in rrf_scores.items():
            result = content_map[doc_id]
            merged_result = RetrievalResult(
                id=doc_id,
                content=result.content,
                score=rrf_score,
                metadata={
                    **result.metadata,
                    'merge_strategy': 'rrf',
                    'rrf_k': k,
                    'original_score': result.score
                }
            )
            merged_results.append(merged_result)
        
        # Sort by RRF score
        merged_results.sort(key=lambda x: x.score, reverse=True)
        
        return merged_results
    
    @staticmethod
    def normalized_merge(
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        vector_weight: float = 0.5,
        graph_weight: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Merge with score normalization
        
        Args:
            vector_results: Results from vector search
            graph_results: Results from graph search
            vector_weight: Weight for vector results
            graph_weight: Weight for graph results
            
        Returns:
            Merged results with normalized scores
        """
        def normalize_scores(results: List[RetrievalResult]) -> Dict[str, float]:
            """Normalize scores to [0, 1] range"""
            if not results:
                return {}
            
            scores = [r.score for r in results]
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score == min_score:
                return {r.id: 0.5 for r in results}
            
            normalized = {}
            for result in results:
                norm_score = (result.score - min_score) / (max_score - min_score)
                normalized[result.id] = norm_score
            
            return normalized
        
        # Normalize scores
        vector_norm = normalize_scores(vector_results)
        graph_norm = normalize_scores(graph_results)
        
        # Combine normalized scores
        all_ids = set(vector_norm.keys()) | set(graph_norm.keys())
        content_map = {r.id: r for r in vector_results + graph_results}
        
        merged_results = []
        for doc_id in all_ids:
            v_score = vector_norm.get(doc_id, 0) * vector_weight
            g_score = graph_norm.get(doc_id, 0) * graph_weight
            combined_score = v_score + g_score
            
            result = content_map[doc_id]
            merged_result = RetrievalResult(
                id=doc_id,
                content=result.content,
                score=combined_score,
                metadata={
                    **result.metadata,
                    'merge_strategy': 'normalized',
                    'vector_norm_score': vector_norm.get(doc_id, 0),
                    'graph_norm_score': graph_norm.get(doc_id, 0)
                }
            )
            merged_results.append(merged_result)
        
        # Sort by combined score
        merged_results.sort(key=lambda x: x.score, reverse=True)
        
        return merged_results
    
    @staticmethod
    def apply_mmr(
        results: List[RetrievalResult],
        lambda_param: float = 0.7,
        similarity_threshold: float = 0.8
    ) -> List[RetrievalResult]:
        """
        Apply Maximal Marginal Relevance (MMR) to diversify results
        
        Args:
            results: Initial retrieval results
            lambda_param: Trade-off between relevance and diversity
            similarity_threshold: Threshold for considering documents similar
            
        Returns:
            Diversified results using MMR
        """
        if len(results) <= 1:
            return results
        
        # Start with highest scoring result
        selected = [results[0]]
        remaining = results[1:]
        
        while remaining and len(selected) < len(results):
            best_score = -1
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                # Calculate relevance score
                relevance = candidate.score
                
                # Calculate diversity (minimum similarity to selected)
                min_similarity = 1.0
                for selected_result in selected:
                    similarity = MergeStrategies._calculate_similarity(
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
        
        # Update metadata
        for i, result in enumerate(selected):
            result.metadata['mmr_rank'] = i + 1
            result.metadata['mmr_lambda'] = lambda_param
        
        return selected
    
    @staticmethod
    def _calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculate text similarity using Jaccard coefficient
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple word-based Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0