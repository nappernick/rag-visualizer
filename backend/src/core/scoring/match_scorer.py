"""
Independent match scoring system for RAG retrieval
Provides consistent, meaningful match percentages regardless of fusion settings
"""
import re
from typing import Dict, List, Optional
from difflib import SequenceMatcher
import math


class MatchScorer:
    """Calculate independent match scores for retrieval results"""
    
    @staticmethod
    def calculate_vector_match_score(
        query: str,
        content: str,
        cosine_similarity: float,
        metadata: Optional[Dict] = None
    ) -> float:
        """
        Calculate match score for vector search results
        Returns a percentage (0-100) representing match quality
        """
        # Base score from cosine similarity (typically 0-1)
        # Convert to percentage and apply sigmoid for better distribution
        base_score = MatchScorer._sigmoid_scale(cosine_similarity) * 100
        
        # Bonus for exact query terms in content
        query_terms = set(query.lower().split())
        content_lower = content.lower()
        
        # Count how many query terms appear in content
        term_matches = sum(1 for term in query_terms if term in content_lower)
        term_coverage = (term_matches / len(query_terms)) if query_terms else 0
        
        # Bonus for term coverage (up to 20% bonus)
        term_bonus = term_coverage * 20
        
        # Bonus for exact phrase match (up to 30% bonus)
        phrase_bonus = 0
        if query.lower() in content_lower:
            phrase_bonus = 30
        
        # Combine scores (max 100)
        final_score = min(100, base_score + term_bonus + phrase_bonus)
        
        return round(final_score, 1)
    
    @staticmethod
    def calculate_graph_match_score(
        query: str,
        entity_name: str,
        entity_type: str,
        match_type: str,
        confidence: float = 1.0,
        path_length: int = 0
    ) -> float:
        """
        Calculate match score for graph search results
        Returns a percentage (0-100) representing match quality
        """
        query_lower = query.lower()
        entity_lower = entity_name.lower()
        
        # Base score based on match type
        if match_type == 'exact' or query_lower == entity_lower:
            base_score = 95
        elif match_type == 'contains' or entity_lower in query_lower or query_lower in entity_lower:
            # Calculate containment ratio
            if entity_lower in query_lower:
                containment_ratio = len(entity_lower) / len(query_lower)
                base_score = 70 + (containment_ratio * 25)  # 70-95 based on how much of query is matched
            elif query_lower in entity_lower:
                containment_ratio = len(query_lower) / len(entity_lower)
                base_score = 60 + (containment_ratio * 30)  # 60-90 based on coverage
            else:
                base_score = 75
        elif match_type == 'fuzzy':
            # Use sequence matching for fuzzy matches
            similarity = SequenceMatcher(None, query_lower, entity_lower).ratio()
            base_score = similarity * 85  # Up to 85% for fuzzy matches
        elif match_type == 'traversal':
            # Traversal matches get lower base scores
            base_score = 40 - (path_length * 10)  # Reduce by 10% per hop
            base_score = max(10, base_score)  # Minimum 10%
        else:
            # Default/unknown match types
            base_score = 30
        
        # Apply confidence modifier
        final_score = base_score * confidence
        
        # Word-level matching bonus
        query_words = set(query_lower.split())
        entity_words = set(entity_lower.split())
        
        if query_words and entity_words:
            word_overlap = len(query_words & entity_words) / len(query_words)
            word_bonus = word_overlap * 15  # Up to 15% bonus for word matches
            final_score = min(100, final_score + word_bonus)
        
        return round(final_score, 1)
    
    @staticmethod
    def calculate_hybrid_match_score(
        query: str,
        vector_score: Optional[float] = None,
        graph_score: Optional[float] = None,
        source: str = "hybrid"
    ) -> float:
        """
        Calculate combined match score for hybrid results
        Returns a percentage (0-100) representing overall match quality
        """
        if source == "vector" and vector_score is not None:
            return vector_score
        elif source == "graph" and graph_score is not None:
            return graph_score
        elif vector_score is not None and graph_score is not None:
            # For true hybrid results, take the maximum
            # (if it matched well in either system, it's a good match)
            return max(vector_score, graph_score)
        else:
            # Fallback
            return 50.0
    
    @staticmethod
    def _sigmoid_scale(x: float, steepness: float = 5.0) -> float:
        """
        Apply sigmoid scaling to spread out scores better
        Maps [0,1] to [0,1] with better distribution
        """
        # Shift and scale for better distribution
        # This makes 0.7 cosine similarity ~ 50%, 0.8 ~ 70%, 0.9 ~ 90%
        adjusted = (x - 0.5) * steepness
        return 1 / (1 + math.exp(-adjusted))
    
    @staticmethod
    def explain_score(
        score: float,
        source: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Provide human-readable explanation of the match score
        """
        if score >= 90:
            quality = "Excellent match"
        elif score >= 75:
            quality = "Strong match"
        elif score >= 60:
            quality = "Good match"
        elif score >= 40:
            quality = "Moderate match"
        elif score >= 25:
            quality = "Weak match"
        else:
            quality = "Poor match"
        
        explanation = f"{quality} ({score:.1f}%)"
        
        if metadata:
            if source == "graph" and "match_type" in metadata:
                match_type = metadata["match_type"]
                if match_type == "exact":
                    explanation += " - Exact entity match"
                elif match_type == "contains":
                    explanation += " - Partial entity match"
                elif match_type == "traversal":
                    path_length = metadata.get("path_length", 0)
                    explanation += f" - Related entity ({path_length} hop{'s' if path_length != 1 else ''})"
            elif source == "vector":
                explanation += " - Semantic similarity"
        
        return explanation