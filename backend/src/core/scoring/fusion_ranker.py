"""
Fusion ranking system that combines vector and graph results
Keeps match scores independent while allowing weighted ranking
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class RankedResult:
    """A result with both match score and ranking score"""
    id: str
    content: str
    match_score: float  # Independent match quality (0-100)
    ranking_score: float  # Weighted score for ordering
    source: str  # "vector", "graph", or "hybrid"
    metadata: Dict
    
    @property
    def display_score(self) -> float:
        """The score shown to users (match score)"""
        return self.match_score


class FusionRanker:
    """
    Handles fusion of vector and graph results with user-controlled weighting
    
    The key insight: 
    - Match scores tell you HOW WELL something matches (quality)
    - Ranking scores tell you WHERE it appears in results (priority)
    - The slider controls priority, not quality
    """
    
    @staticmethod
    def rank_results(
        vector_results: List[Dict],
        graph_results: List[Dict],
        vector_weight: float = 0.5,  # 0 = all graph, 1 = all vector
        strategy: str = "hybrid",
        deduplication: str = "merge"  # "merge", "prefer_vector", "prefer_graph"
    ) -> List[RankedResult]:
        """
        Combine and rank results based on fusion weights
        
        Args:
            vector_results: Results from vector search with match_scores
            graph_results: Results from graph search with match_scores
            vector_weight: Weight for vector results (0-1)
            strategy: "vector", "graph", or "hybrid"
            deduplication: How to handle duplicate results
            
        Returns:
            Ranked list of results with independent match scores
        """
        graph_weight = 1.0 - vector_weight
        ranked_results = []
        seen_content = {}  # For deduplication
        
        # Process vector results
        if strategy in ["vector", "hybrid"]:
            for i, result in enumerate(vector_results):
                # Position penalty - results further down get lower ranking
                position_factor = 1.0 / (1.0 + i * 0.1)
                
                # Calculate ranking score (for ordering)
                ranking_score = (
                    result['match_score'] * 
                    vector_weight * 
                    position_factor
                )
                
                # Check for duplicates
                content_key = result['content'][:100].lower()  # First 100 chars
                if content_key in seen_content:
                    if deduplication == "merge":
                        # Merge with existing result
                        existing = seen_content[content_key]
                        # Take the higher match score
                        existing.match_score = max(existing.match_score, result['match_score'])
                        # Boost ranking score for appearing in both
                        existing.ranking_score += ranking_score * 0.5
                        continue
                    elif deduplication == "prefer_graph":
                        continue  # Skip vector duplicate
                    # else prefer_vector - will replace the graph result
                
                ranked_result = RankedResult(
                    id=result['id'],
                    content=result['content'],
                    match_score=result['match_score'],  # Independent score
                    ranking_score=ranking_score,  # Weighted for ranking
                    source="vector",
                    metadata={
                        **result.get('metadata', {}),
                        'vector_weight': vector_weight,
                        'position': i,
                        'ranking_factors': {
                            'base_score': result['match_score'],
                            'weight': vector_weight,
                            'position_factor': position_factor
                        }
                    }
                )
                
                ranked_results.append(ranked_result)
                seen_content[content_key] = ranked_result
        
        # Process graph results
        if strategy in ["graph", "hybrid"]:
            for i, result in enumerate(graph_results):
                # Position penalty
                position_factor = 1.0 / (1.0 + i * 0.1)
                
                # Graph results get bonus for relationships
                relationship_bonus = 1.0
                if result.get('metadata', {}).get('path_length', 0) == 0:
                    relationship_bonus = 1.2  # Direct match bonus
                
                # Calculate ranking score
                ranking_score = (
                    result['match_score'] * 
                    graph_weight * 
                    position_factor * 
                    relationship_bonus
                )
                
                # Check for duplicates
                content_key = result['content'][:100].lower()
                if content_key in seen_content:
                    if deduplication == "merge":
                        existing = seen_content[content_key]
                        existing.match_score = max(existing.match_score, result['match_score'])
                        existing.ranking_score += ranking_score * 0.5
                        existing.source = "hybrid"  # Mark as found by both
                        continue
                    elif deduplication == "prefer_vector":
                        continue
                
                ranked_result = RankedResult(
                    id=result['id'],
                    content=result['content'],
                    match_score=result['match_score'],
                    ranking_score=ranking_score,
                    source="graph",
                    metadata={
                        **result.get('metadata', {}),
                        'graph_weight': graph_weight,
                        'position': i,
                        'ranking_factors': {
                            'base_score': result['match_score'],
                            'weight': graph_weight,
                            'position_factor': position_factor,
                            'relationship_bonus': relationship_bonus
                        }
                    }
                )
                
                ranked_results.append(ranked_result)
                seen_content[content_key] = ranked_result
        
        # Sort by ranking score (not match score!)
        ranked_results.sort(key=lambda x: x.ranking_score, reverse=True)
        
        # Add final rank position
        for i, result in enumerate(ranked_results):
            result.metadata['final_rank'] = i + 1
        
        return ranked_results
    
    @staticmethod
    def explain_ranking(result: RankedResult, verbose: bool = False) -> str:
        """
        Explain why a result appears where it does in the ranking
        """
        explanation = []
        
        # Match quality
        explanation.append(f"Match Quality: {result.match_score:.1f}%")
        
        # Source and weighting
        if result.source == "vector":
            weight = result.metadata.get('vector_weight', 0.5)
            explanation.append(f"Source: Vector Search (weight: {weight:.0%})")
        elif result.source == "graph":
            weight = result.metadata.get('graph_weight', 0.5)
            explanation.append(f"Source: Graph Search (weight: {weight:.0%})")
        else:
            explanation.append("Source: Hybrid (found by both)")
        
        # Ranking position
        rank = result.metadata.get('final_rank', 0)
        if rank:
            explanation.append(f"Rank: #{rank}")
        
        if verbose and 'ranking_factors' in result.metadata:
            factors = result.metadata['ranking_factors']
            explanation.append("Ranking factors:")
            for key, value in factors.items():
                explanation.append(f"  - {key}: {value:.2f}")
        
        return " | ".join(explanation)
    
    @staticmethod
    def get_weight_recommendation(query_type: str) -> Tuple[float, str]:
        """
        Recommend optimal weights based on query type
        
        Returns:
            (vector_weight, explanation)
        """
        recommendations = {
            "factual": (0.3, "Graph search excels at finding specific facts and entities"),
            "conceptual": (0.7, "Vector search better understands abstract concepts"),
            "navigational": (0.2, "Graph search directly finds named entities"),
            "analytical": (0.5, "Balanced approach for analytical queries"),
            "exploratory": (0.6, "Vector search helps discover related content"),
        }
        
        return recommendations.get(query_type, (0.5, "Balanced fusion for general queries"))
    
    @staticmethod
    def analyze_results_distribution(results: List[RankedResult]) -> Dict:
        """
        Analyze the distribution of results by source
        Useful for understanding how the fusion is working
        """
        total = len(results)
        if total == 0:
            return {}
        
        vector_count = sum(1 for r in results if r.source == "vector")
        graph_count = sum(1 for r in results if r.source == "graph")
        hybrid_count = sum(1 for r in results if r.source == "hybrid")
        
        # Average scores by source
        vector_scores = [r.match_score for r in results if r.source == "vector"]
        graph_scores = [r.match_score for r in results if r.source == "graph"]
        
        return {
            "total_results": total,
            "by_source": {
                "vector": vector_count,
                "graph": graph_count,
                "hybrid": hybrid_count
            },
            "percentages": {
                "vector": vector_count / total * 100,
                "graph": graph_count / total * 100,
                "hybrid": hybrid_count / total * 100
            },
            "average_match_scores": {
                "vector": sum(vector_scores) / len(vector_scores) if vector_scores else 0,
                "graph": sum(graph_scores) / len(graph_scores) if graph_scores else 0,
                "overall": sum(r.match_score for r in results) / total
            },
            "top_source": max(
                ["vector", "graph", "hybrid"],
                key=lambda s: sum(1 for r in results[:3] if r.source == s)
            )
        }