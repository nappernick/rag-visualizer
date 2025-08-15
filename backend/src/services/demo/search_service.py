"""
Demo Search Service - Handles enhanced search functionality
"""
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import asyncio

from ...core.retrieval.fusion_controller import FusionController
from ...core.retrieval.hybrid_search import HybridSearch
from ...core.query.query_enhancer import QueryEnhancer
from ...services.vector_service import VectorService
from ...services.graph_service import GraphService

logger = logging.getLogger(__name__)


class DemoSearchService:
    """Service for handling enhanced search operations in demo mode"""
    
    def __init__(self, 
                 fusion_controller: FusionController,
                 vector_service: VectorService,
                 graph_service: GraphService,
                 bedrock_client=None):
        self.fusion_controller = fusion_controller
        self.vector_service = vector_service
        self.graph_service = graph_service
        self.bedrock_client = bedrock_client
        self.query_enhancer = QueryEnhancer()
        self.hybrid_search = HybridSearch(
            vector_service=vector_service,
            graph_service=graph_service,
            enable_caching=True
        )
    
    async def enhanced_search(
        self,
        query: str,
        use_smart_routing: bool = True,
        include_explanation: bool = True,
        max_results: int = 10,
        strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform enhanced search with smart routing and explanations
        
        Args:
            query: The search query
            use_smart_routing: Whether to automatically determine search strategy
            include_explanation: Whether to include search explanation
            max_results: Maximum number of results
            strategy: Override strategy (vector, graph, hybrid)
            
        Returns:
            Search results with metadata and explanations
        """
        start_time = datetime.now()
        
        # Determine search strategy
        if use_smart_routing and not strategy:
            strategy = await self._determine_smart_strategy(query)
        elif not strategy:
            strategy = "hybrid"
        
        # Get query embedding
        query_embedding = await self.vector_service.get_embedding(query)
        
        # Perform search based on strategy
        results = []
        if strategy == "vector":
            results = await self._vector_search(query, query_embedding, max_results)
        elif strategy == "graph":
            results = await self._graph_search(query, query_embedding, max_results)
        else:  # hybrid
            results = await self._hybrid_search(query, query_embedding, max_results)
        
        # Generate explanation if requested
        explanation = None
        if include_explanation and self.bedrock_client:
            explanation = await self._generate_explanation(
                query, results[:3], strategy
            )
        
        # Extract highlights
        highlights = self._extract_highlights(query, [r.get('content', '') for r in results[:5]])
        
        # Calculate search time
        search_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "results": results,
            "strategy": strategy,
            "total_results": len(results),
            "search_time": search_time,
            "explanation": explanation,
            "highlights": highlights,
            "metadata": {
                "query_length": len(query.split()),
                "embedding_generated": query_embedding is not None,
                "smart_routing_used": use_smart_routing
            }
        }
    
    async def _vector_search(self, query: str, query_embedding: List[float], max_results: int) -> List[Dict]:
        """Perform vector-only search"""
        results = await self.vector_service.search(
            query_embedding=query_embedding,
            top_k=max_results
        )
        
        return [
            {
                "id": r.id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
                "source": "vector"
            }
            for r in results
        ]
    
    async def _graph_search(self, query: str, query_embedding: List[float], max_results: int) -> List[Dict]:
        """Perform graph-only search"""
        # Extract entities from query
        entities = await self.graph_service.extract_entities_from_text(query)
        
        if not entities:
            return []
        
        # Search for related entities and content
        results = []
        for entity in entities[:3]:  # Limit to top 3 entities
            entity_results = await self.graph_service.find_related_content(
                entity_name=entity['name'],
                max_results=max_results // 3
            )
            results.extend(entity_results)
        
        # Deduplicate and format
        seen = set()
        unique_results = []
        for r in results:
            if r.get('id') not in seen:
                seen.add(r.get('id'))
                unique_results.append({
                    "id": r.get('id'),
                    "content": r.get('content'),
                    "score": r.get('score', 0.5),
                    "metadata": r.get('metadata', {}),
                    "source": "graph"
                })
        
        return unique_results[:max_results]
    
    async def _hybrid_search(self, query: str, query_embedding: List[float], max_results: int) -> List[Dict]:
        """Perform hybrid search combining vector and graph"""
        # Run both searches in parallel
        vector_task = self._vector_search(query, query_embedding, max_results)
        graph_task = self._graph_search(query, query_embedding, max_results)
        
        vector_results, graph_results = await asyncio.gather(vector_task, graph_task)
        
        # Merge results using reciprocal rank fusion
        merged = self._merge_results(vector_results, graph_results)
        
        return merged[:max_results]
    
    def _merge_results(self, vector_results: List[Dict], graph_results: List[Dict]) -> List[Dict]:
        """Merge vector and graph results using reciprocal rank fusion"""
        k = 60  # RRF constant
        scores = {}
        
        # Score vector results
        for i, result in enumerate(vector_results):
            doc_id = result['id']
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + i + 1)
            
        # Score graph results  
        for i, result in enumerate(graph_results):
            doc_id = result['id']
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + i + 1)
        
        # Combine results
        all_results = {r['id']: r for r in vector_results + graph_results}
        
        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        return [
            {**all_results[doc_id], "score": scores[doc_id]}
            for doc_id in sorted_ids
        ]
    
    async def _determine_smart_strategy(self, query: str) -> str:
        """Determine the best search strategy based on query characteristics"""
        if not self.bedrock_client:
            return "hybrid"
        
        # Simple heuristics first
        query_lower = query.lower()
        
        # Graph-oriented queries
        if any(word in query_lower for word in ['relationship', 'connection', 'related', 'between', 'who', 'what entity']):
            return "graph"
        
        # Vector-oriented queries  
        if any(word in query_lower for word in ['similar', 'like', 'about', 'describe', 'explain']):
            return "vector"
        
        # Default to hybrid for complex queries
        if len(query.split()) > 10:
            return "hybrid"
        
        return "hybrid"
    
    async def _generate_explanation(
        self,
        query: str,
        results: List[Dict],
        strategy: str
    ) -> str:
        """Generate an explanation of the search results"""
        if not self.bedrock_client:
            return f"Search performed using {strategy} strategy"
        
        try:
            prompt = f"""
            Explain why these search results are relevant to the query.
            
            Query: {query}
            Strategy: {strategy}
            
            Top Results:
            {chr(10).join([f"- {r.get('content', '')[:200]}..." for r in results])}
            
            Provide a brief explanation (2-3 sentences) of why these results match the query.
            """
            
            response = self.bedrock_client.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body={
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.3
                }
            )
            
            return response.get('content', [{}])[0].get('text', '')
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return f"Results found using {strategy} strategy"
    
    def _extract_highlights(self, query: str, contents: List[str]) -> List[str]:
        """Extract relevant highlights from content"""
        query_words = set(query.lower().split())
        highlights = []
        
        for content in contents:
            sentences = content.split('.')
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in query_words):
                    highlights.append(sentence.strip())
                    if len(highlights) >= 3:
                        return highlights
        
        return highlights