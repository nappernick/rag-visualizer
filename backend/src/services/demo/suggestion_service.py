"""
Suggestion Service - Handles query suggestions and recommendations
"""
from typing import List, Dict, Any
import logging
from collections import Counter
import re

logger = logging.getLogger(__name__)


class SuggestionService:
    """Service for generating query suggestions and recommendations"""
    
    def __init__(self, bedrock_client=None):
        self.bedrock_client = bedrock_client
        self.common_patterns = [
            "What is {topic}?",
            "How does {topic} work?",
            "Compare {topic1} and {topic2}",
            "Explain the relationship between {topic1} and {topic2}",
            "What are the benefits of {topic}?",
            "Find examples of {topic}",
            "Show me {topic} implementations",
            "What are the challenges with {topic}?"
        ]
    
    async def get_suggestions(
        self,
        partial_query: str,
        context: List[str] = None,
        max_suggestions: int = 5
    ) -> List[str]:
        """
        Generate query suggestions based on partial input
        
        Args:
            partial_query: The partial query entered by user
            context: Recent queries or document context
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of suggested queries
        """
        suggestions = []
        
        # Get keyword-based suggestions
        keyword_suggestions = self._get_keyword_suggestions(partial_query)
        suggestions.extend(keyword_suggestions)
        
        # Get pattern-based suggestions
        pattern_suggestions = self._get_pattern_suggestions(partial_query)
        suggestions.extend(pattern_suggestions)
        
        # Get AI-powered suggestions if available
        if self.bedrock_client and len(suggestions) < max_suggestions:
            ai_suggestions = await self._get_ai_suggestions(partial_query, context)
            suggestions.extend(ai_suggestions)
        
        # Deduplicate and limit
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen and s != partial_query:
                seen.add(s)
                unique_suggestions.append(s)
                if len(unique_suggestions) >= max_suggestions:
                    break
        
        return unique_suggestions
    
    def _get_keyword_suggestions(self, partial_query: str) -> List[str]:
        """Generate suggestions based on keywords"""
        suggestions = []
        query_lower = partial_query.lower()
        
        # Common completions based on starting words
        if query_lower.startswith("what"):
            suggestions.extend([
                f"{partial_query} is the difference between",
                f"{partial_query} are the benefits of",
                f"{partial_query} is the purpose of"
            ])
        elif query_lower.startswith("how"):
            suggestions.extend([
                f"{partial_query} does it work",
                f"{partial_query} to implement",
                f"{partial_query} can I use"
            ])
        elif query_lower.startswith("why"):
            suggestions.extend([
                f"{partial_query} is it important",
                f"{partial_query} should I use",
                f"{partial_query} does it matter"
            ])
        elif query_lower.startswith("find"):
            suggestions.extend([
                f"{partial_query} examples of",
                f"{partial_query} information about",
                f"{partial_query} all references to"
            ])
        elif query_lower.startswith("show"):
            suggestions.extend([
                f"{partial_query} me examples",
                f"{partial_query} the relationship",
                f"{partial_query} all instances"
            ])
        
        return suggestions[:3]
    
    def _get_pattern_suggestions(self, partial_query: str) -> List[str]:
        """Generate suggestions based on common patterns"""
        suggestions = []
        
        # Extract potential topics from the query
        words = partial_query.split()
        nouns = [w for w in words if len(w) > 3 and w[0].isupper() or w.lower() in ['api', 'data', 'model', 'system']]
        
        if nouns:
            main_topic = nouns[0]
            for pattern in self.common_patterns[:3]:
                if "{topic}" in pattern:
                    suggestions.append(pattern.replace("{topic}", main_topic))
                elif "{topic1}" in pattern and len(nouns) > 1:
                    suggestions.append(
                        pattern.replace("{topic1}", nouns[0])
                               .replace("{topic2}", nouns[1])
                    )
        
        return suggestions
    
    async def _get_ai_suggestions(
        self,
        partial_query: str,
        context: List[str] = None
    ) -> List[str]:
        """Generate AI-powered suggestions"""
        if not self.bedrock_client:
            return []
        
        try:
            context_str = "\n".join(context) if context else "No context available"
            
            prompt = f"""
            Generate 3 query suggestions to complete or expand this partial query:
            Partial Query: "{partial_query}"
            
            Context (recent queries or topics):
            {context_str}
            
            Provide 3 relevant, specific query suggestions that would help the user.
            Format as a simple list, one per line.
            """
            
            response = self.bedrock_client.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body={
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.7
                }
            )
            
            text = response.get('content', [{}])[0].get('text', '')
            suggestions = [s.strip() for s in text.split('\n') if s.strip()]
            return suggestions[:3]
            
        except Exception as e:
            logger.error(f"Error generating AI suggestions: {e}")
            return []
    
    def get_simple_suggestions(self, query: str) -> List[str]:
        """Get simple fallback suggestions when AI is not available"""
        suggestions = []
        query_lower = query.lower()
        
        # Extract key terms
        terms = re.findall(r'\b[a-z]+\b', query_lower)
        
        if not terms:
            return [
                "What is RAG?",
                "How does vector search work?",
                "Explain knowledge graphs",
                "Show me all documents",
                "Find similar content"
            ]
        
        main_term = max(terms, key=len) if terms else "topic"
        
        return [
            f"What is {main_term}?",
            f"How does {main_term} work?",
            f"Examples of {main_term}",
            f"Benefits of {main_term}",
            f"{main_term} best practices"
        ]