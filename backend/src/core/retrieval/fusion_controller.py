"""
Tunable Fusion Controller for dynamic RAG retrieval strategy
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
import yaml
from pathlib import Path
import numpy as np
from sentence_transformers import CrossEncoder
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ...models import RetrievalResult
from .vector_retriever import VectorRetriever
from .graph_retriever import GraphRetriever
from .vector_node_mapper import VectorNodeMapper
from .hybrid_search import EnhancedHybridSearch, SearchConfig
from ..query.query_enhancer import QueryEnhancer, QueryType
from ..rag.graph_rag import GraphRAG
from ...services.id_mapper import IDMapper
from ..temporal.temporal_utils import get_temporal_score, apply_temporal_boost, detect_doc_type

logger = logging.getLogger(__name__)


class FusionController:
    """
    Controls and orchestrates fusion of vector and graph retrieval with tunable parameters.
    """
    
    def __init__(self, config_path: Optional[str] = None, bedrock_client=None, graph_store=None):
        # Load fusion configuration
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path(__file__).parent.parent.parent.parent / "config" / "fusion_config.yaml"
        
        self.load_config()
        
        # Initialize retrievers
        self.vector_retriever = VectorRetriever(self.base_config)
        self.graph_retriever = GraphRetriever(self.base_config)
        self.vector_node_mapper = VectorNodeMapper(self.base_config)
        
        # Initialize new enhanced components
        self.hybrid_search = EnhancedHybridSearch(SearchConfig(
            vector_weight=self.fusion_config.get('vector_weight', 0.4),
            keyword_weight=self.fusion_config.get('keyword_weight', 0.3),
            metadata_weight=self.fusion_config.get('metadata_weight', 0.3)
        ))
        
        self.query_enhancer = QueryEnhancer(bedrock_client=bedrock_client)
        
        # Initialize GraphRAG if graph store is available
        self.graph_rag = None
        if graph_store:
            self.id_mapper = IDMapper()
            self.graph_rag = GraphRAG(
                graph_store=graph_store,
                vector_retriever=self.vector_retriever,
                id_mapper=self.id_mapper,
                bedrock_client=bedrock_client
            )
            logger.info("GraphRAG initialized for multi-hop reasoning")
        
        # Initialize reranker if enabled
        self._reranker = None
        if self.fusion_config.get('use_reranker'):
            try:
                model_name = self.fusion_config.get('reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
                self._reranker = CrossEncoder(model_name)
                logger.info(f"Loaded reranking model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}")
        
        # Thread pool for parallel retrieval
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.fusion_config = config.get('fusion', {})
        self.presets = config.get('presets', {})
        
        # Base configuration for retrievers
        self.base_config = {
            'vector_store': {
                'provider': 'qdrant',
                'host': 'localhost',
                'port': 6333,
                'collection_name': 'rag_chunks',
                'embedding_dim': 1536
            },
            'graph_store': {
                'provider': 'neo4j',
                'uri': 'bolt://localhost:7687',
                'username': 'neo4j',
                'password': 'password'
            },
            'retrieval': self.fusion_config
        }
    
    def update_config(self, overrides: Dict[str, Any]):
        """Update fusion configuration with overrides"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.fusion_config, overrides)
        logger.info(f"Updated fusion config with overrides: {overrides}")
    
    def apply_preset(self, preset_name: str):
        """Apply a preset configuration"""
        if preset_name in self.presets:
            preset_config = self.presets[preset_name]
            self.update_config(preset_config)
            logger.info(f"Applied preset: {preset_name}")
        else:
            logger.warning(f"Preset not found: {preset_name}")
    
    async def retrieve_enhanced(
        self,
        query: str,
        query_embedding: List[float],
        use_query_enhancement: bool = True,
        use_graph_rag: bool = False,
        preset: Optional[str] = None,
        **overrides
    ) -> Dict[str, Any]:
        """
        Enhanced retrieval with query decomposition, hybrid search, and GraphRAG.
        
        Args:
            query: Text query
            query_embedding: Query embedding vector
            use_query_enhancement: Whether to decompose and expand query
            use_graph_rag: Whether to use GraphRAG for multi-hop reasoning
            preset: Optional preset name to apply
            **overrides: Parameter overrides
            
        Returns:
            Enhanced retrieval results with reasoning
        """
        # Apply preset if specified
        if preset:
            self.apply_preset(preset)
        
        # Step 1: Enhance query if enabled
        enhanced_query = None
        if use_query_enhancement:
            enhanced_query = await self.query_enhancer.enhance_query(query)
            logger.info(f"Query enhanced - Type: {enhanced_query.query_type}, "
                       f"Sub-queries: {len(enhanced_query.sub_queries)}")
        
        # Step 2: Determine retrieval strategy
        if use_graph_rag and self.graph_rag and self._needs_multi_hop(query, enhanced_query):
            # Use GraphRAG for multi-hop reasoning
            logger.info("Using GraphRAG for multi-hop reasoning")
            result = await self.graph_rag.answer_query(
                query=query,
                query_embedding=query_embedding,
                max_hops=self.fusion_config.get('graph_expansion_depth', 3)
            )
            return result
        
        # Step 3: Use enhanced hybrid search
        search_config = SearchConfig(
            vector_weight=self.fusion_config.get('vector_weight', 0.4),
            keyword_weight=self.fusion_config.get('keyword_weight', 0.3),
            metadata_weight=self.fusion_config.get('metadata_weight', 0.3),
            vector_top_k=self.fusion_config.get('vector_top_k', 20),
            keyword_top_k=self.fusion_config.get('keyword_top_k', 20),
            use_parallel=True
        )
        
        # If query was enhanced, search for each sub-query
        if enhanced_query and len(enhanced_query.sub_queries) > 1:
            all_results = []
            for sub_query in enhanced_query.sub_queries:
                sub_results = await self.hybrid_search.search(
                    query=sub_query.question,
                    query_embedding=query_embedding,  # Could generate separate embeddings
                    config_override=search_config
                )
                all_results.extend(sub_results)
            
            # Deduplicate and re-rank
            results = self._deduplicate_results(all_results)
        else:
            # Single query search
            results = await self.hybrid_search.search(
                query=query,
                query_embedding=query_embedding,
                config_override=search_config
            )
        
        # Step 4: Apply temporal scoring
        results = self._apply_temporal_scoring(results, query)
        
        # Step 5: Apply reranking if enabled
        if self.fusion_config.get('use_reranker') and self._reranker:
            results = self._rerank_results(results, query)
        
        # Step 6: Optimize for context budget
        results = self._optimize_context(results)
        
        # Return results with metadata
        return {
            'results': results[:self.fusion_config.get('final_top_k', 10)],
            'enhanced_query': enhanced_query.__dict__ if enhanced_query else None,
            'strategy_used': 'graph_rag' if use_graph_rag else 'hybrid_search',
            'total_retrieved': len(results)
        }
    
    def _needs_multi_hop(self, query: str, enhanced_query) -> bool:
        """Determine if query requires multi-hop reasoning"""
        if enhanced_query and enhanced_query.query_type == QueryType.MULTI_HOP:
            return True
        
        # Check for multi-hop indicators
        indicators = ['affect', 'cause', 'lead to', 'result in', 'impact', 'influence']
        return any(ind in query.lower() for ind in indicators)
    
    def _deduplicate_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Deduplicate results while preserving best scores"""
        seen = {}
        for result in results:
            if result.chunk_id not in seen or result.score > seen[result.chunk_id].score:
                seen[result.chunk_id] = result
        
        deduplicated = list(seen.values())
        deduplicated.sort(key=lambda x: x.score, reverse=True)
        return deduplicated
    
    def _apply_temporal_scoring(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Apply temporal decay scoring to results based on document age and type"""
        # Determine temporal weight based on query
        temporal_weight = apply_temporal_boost(query)
        semantic_weight = 1.0 - temporal_weight
        
        for result in results:
            # Get temporal metadata
            created_at_ms = result.metadata.get('created_at_ms')
            doc_type = result.metadata.get('doc_type')
            
            # Auto-detect doc type if not present
            if not doc_type and 'path' in result.metadata:
                doc_type = detect_doc_type(result.metadata['path'], result.content[:500])
                result.metadata['doc_type'] = doc_type
            
            # Calculate temporal score
            temporal_score = get_temporal_score(created_at_ms, doc_type or 'default')
            
            # Combine semantic and temporal scores
            original_score = result.score
            result.score = (semantic_weight * original_score) + (temporal_weight * temporal_score)
            
            # Store scores in metadata for transparency
            result.metadata['semantic_score'] = original_score
            result.metadata['temporal_score'] = temporal_score
            result.metadata['temporal_weight'] = temporal_weight
        
        # Re-sort by new combined scores
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def retrieve(
        self,
        query: str,
        query_embedding: List[float],
        preset: Optional[str] = None,
        **overrides
    ) -> List[RetrievalResult]:
        """
        Main retrieval method with tunable fusion.
        
        Args:
            query: Text query
            query_embedding: Query embedding vector
            preset: Optional preset name to apply
            **overrides: Parameter overrides for this query
            
        Returns:
            Fused and optimized retrieval results
        """
        # Apply preset if specified
        if preset:
            self.apply_preset(preset)
        
        # Apply query-specific overrides
        if overrides:
            original_config = dict(self.fusion_config)
            self.update_config(overrides)
        
        try:
            # Analyze query to determine strategy
            if self.fusion_config.get('auto_strategy'):
                strategy = self._analyze_query_strategy(query)
            else:
                strategy = 'hybrid'
            
            logger.info(f"Using retrieval strategy: {strategy}")
            
            # Perform retrieval based on strategy
            if strategy == 'vector':
                results = self._vector_only_retrieve(query_embedding)
            elif strategy == 'graph':
                results = self._graph_only_retrieve(query, query_embedding)
            else:  # hybrid
                results = self._hybrid_retrieve(query, query_embedding)
            
            # Apply hierarchical tuning
            results = self._apply_hierarchical_tuning(results)
            
            # Apply temporal scoring
            results = self._apply_temporal_scoring(results, query)
            
            # Apply reranking if enabled
            if self.fusion_config.get('use_reranker') and self._reranker:
                results = self._rerank_results(results, query)
            
            # Optimize for context budget
            results = self._optimize_context(results)
            
            # Return top K results
            final_k = self.fusion_config.get('final_top_k', 10)
            return results[:final_k]
            
        finally:
            # Restore original config if overrides were applied
            if overrides:
                self.fusion_config = original_config
    
    def _analyze_query_strategy(self, query: str) -> str:
        """Analyze query to determine optimal retrieval strategy"""
        query_lower = query.lower()
        query_analysis = self.fusion_config.get('query_analysis', {})
        
        # Get keyword lists
        conceptual_keywords = query_analysis.get('conceptual_keywords', [])
        specific_keywords = query_analysis.get('specific_keywords', [])
        
        # Calculate scores
        conceptual_score = sum(1 for kw in conceptual_keywords if kw in query_lower)
        specific_score = sum(1 for kw in specific_keywords if kw in query_lower)
        
        # Query length heuristic
        word_count = len(query.split())
        if word_count > 10:
            specific_score += 2
        elif word_count < 5:
            conceptual_score += 1
        
        # Calculate confidence
        total_keywords = len(conceptual_keywords) + len(specific_keywords)
        confidence = max(conceptual_score, specific_score) / max(total_keywords * 0.1, 1)
        
        # Determine strategy
        force_hybrid_threshold = self.fusion_config.get('force_hybrid_threshold', 0.5)
        
        if confidence < force_hybrid_threshold:
            return 'hybrid'
        elif conceptual_score > specific_score + 2:
            return 'graph'
        elif specific_score > conceptual_score + 2:
            return 'vector'
        else:
            return 'hybrid'
    
    def _vector_only_retrieve(self, query_embedding: List[float]) -> List[RetrievalResult]:
        """Vector-only retrieval"""
        k = self.fusion_config.get('vector_top_k', 20)
        
        # Get vector strategy config
        vector_config = self.fusion_config.get('strategies', {}).get('vector', {})
        
        # Perform retrieval
        results = self.vector_retriever.retrieve(query_embedding, k)
        
        # Apply MMR if enabled for diversity
        if vector_config.get('use_mmr'):
            results = self._apply_mmr(results, vector_config.get('mmr_lambda', 0.7))
        
        # Filter by similarity threshold
        threshold = vector_config.get('similarity_threshold', 0.7)
        results = [r for r in results if r.score >= threshold]
        
        return results
    
    def _graph_only_retrieve(self, query: str, query_embedding: List[float]) -> List[RetrievalResult]:
        """Graph-only retrieval using vector seeds"""
        # Get seed nodes from vector search
        seed_k = min(5, self.fusion_config.get('vector_top_k', 20) // 4)
        seed_nodes = self.vector_node_mapper.get_seed_nodes_from_search(query_embedding, seed_k)
        
        # Get graph strategy config
        graph_config = self.fusion_config.get('strategies', {}).get('graph', {})
        
        # Perform graph retrieval from seeds
        results = self.graph_retriever.retrieve_from_seeds(
            seed_nodes,
            query=query,
            max_depth=self.fusion_config.get('graph_expansion_depth', 2),
            confidence_threshold=self.fusion_config.get('graph_confidence_threshold', 0.6),
            max_nodes=graph_config.get('max_traversal_nodes', 100)
        )
        
        # Limit to configured top K
        k = self.fusion_config.get('graph_top_k', 15)
        return results[:k]
    
    def _hybrid_retrieve(self, query: str, query_embedding: List[float]) -> List[RetrievalResult]:
        """Hybrid retrieval combining vector and graph"""
        # Configure parallel retrieval if enabled
        performance_config = self.fusion_config.get('performance', {})
        use_parallel = performance_config.get('parallel_retrieval', True)
        
        if use_parallel:
            # Run retrievals in parallel
            vector_future = self.executor.submit(
                self._vector_only_retrieve, query_embedding
            )
            graph_future = self.executor.submit(
                self._graph_only_retrieve, query, query_embedding
            )
            
            vector_results = vector_future.result()
            graph_results = graph_future.result()
        else:
            # Sequential retrieval
            vector_results = self._vector_only_retrieve(query_embedding)
            graph_results = self._graph_only_retrieve(query, query_embedding)
        
        # Merge results based on strategy
        hybrid_config = self.fusion_config.get('strategies', {}).get('hybrid', {})
        merge_strategy = hybrid_config.get('merge_strategy', 'weighted')
        
        if merge_strategy == 'reciprocal_rank':
            return self._merge_reciprocal_rank(vector_results, graph_results)
        elif merge_strategy == 'normalized':
            return self._merge_normalized(vector_results, graph_results)
        else:  # weighted
            return self._merge_weighted(vector_results, graph_results)
    
    def _merge_weighted(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Merge results using weighted combination"""
        vector_weight = self.fusion_config.get('vector_weight', 0.7)
        graph_weight = 1 - vector_weight
        
        # Handle duplicates
        hybrid_config = self.fusion_config.get('strategies', {}).get('hybrid', {})
        duplicate_handling = hybrid_config.get('duplicate_handling', 'max')
        
        seen_chunks = {}
        
        # Process vector results
        for result in vector_results:
            result.score *= vector_weight
            seen_chunks[result.chunk_id] = result
        
        # Process graph results
        for result in graph_results:
            result.score *= graph_weight
            
            if result.chunk_id in seen_chunks:
                existing = seen_chunks[result.chunk_id]
                
                if duplicate_handling == 'max':
                    existing.score = max(existing.score, result.score)
                elif duplicate_handling == 'average':
                    existing.score = (existing.score + result.score) / 2
                else:  # sum
                    existing.score += result.score
                
                existing.source = 'hybrid'
                existing.metadata.update(result.metadata)
            else:
                seen_chunks[result.chunk_id] = result
        
        # Sort by score
        merged = list(seen_chunks.values())
        merged.sort(key=lambda x: x.score, reverse=True)
        
        return merged
    
    def _merge_reciprocal_rank(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Merge using Reciprocal Rank Fusion"""
        k = 60  # RRF constant
        chunk_scores = {}
        
        # Calculate RRF scores for vector results
        for rank, result in enumerate(vector_results, 1):
            chunk_id = result.chunk_id
            rrf_score = 1 / (k + rank)
            
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {'result': result, 'score': 0}
            
            chunk_scores[chunk_id]['score'] += rrf_score
        
        # Calculate RRF scores for graph results
        for rank, result in enumerate(graph_results, 1):
            chunk_id = result.chunk_id
            rrf_score = 1 / (k + rank)
            
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {'result': result, 'score': 0}
            else:
                # Update to hybrid source
                chunk_scores[chunk_id]['result'].source = 'hybrid'
            
            chunk_scores[chunk_id]['score'] += rrf_score
        
        # Extract and sort results
        merged = []
        for chunk_data in chunk_scores.values():
            result = chunk_data['result']
            result.score = chunk_data['score']
            merged.append(result)
        
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged
    
    def _merge_normalized(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Merge with score normalization"""
        # Normalize vector scores
        if vector_results:
            max_vector = max(r.score for r in vector_results)
            min_vector = min(r.score for r in vector_results)
            range_vector = max_vector - min_vector or 1
            
            for r in vector_results:
                r.score = (r.score - min_vector) / range_vector
        
        # Normalize graph scores
        if graph_results:
            max_graph = max(r.score for r in graph_results)
            min_graph = min(r.score for r in graph_results)
            range_graph = max_graph - min_graph or 1
            
            for r in graph_results:
                r.score = (r.score - min_graph) / range_graph
        
        # Now merge with weighted combination
        return self._merge_weighted(vector_results, graph_results)
    
    def _apply_mmr(self, results: List[RetrievalResult], lambda_param: float = 0.7) -> List[RetrievalResult]:
        """Apply Maximum Marginal Relevance for diversity"""
        if not results:
            return []
        
        # Select first result (highest score)
        selected = [results[0]]
        remaining = results[1:]
        
        while remaining and len(selected) < len(results):
            mmr_scores = []
            
            for candidate in remaining:
                # Relevance score
                relevance = candidate.score
                
                # Maximum similarity to already selected
                max_sim = 0
                for selected_result in selected:
                    # Simple content similarity (could use embeddings for better similarity)
                    sim = self._text_similarity(candidate.content, selected_result.content)
                    max_sim = max(max_sim, sim)
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append((candidate, mmr))
            
            # Select result with highest MMR score
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            best_candidate = mmr_scores[0][0]
            
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        
        return selected
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _apply_hierarchical_tuning(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Apply hierarchical weighting to results"""
        hier_config = self.fusion_config.get('hierarchical', {})
        
        for result in results:
            meta = result.metadata
            
            # Apply summary boost
            if self.fusion_config.get('prioritize_summaries') and meta.get('is_summary'):
                result.score *= self.fusion_config.get('summary_boost', 1.2)
            
            # Apply chunk type weights
            chunk_type = meta.get('chunk_type', 'standard')
            if chunk_type == 'section':
                result.score *= hier_config.get('section_weight', 1.1)
            
            # Apply parent context boost
            if hier_config.get('include_parent_context') and meta.get('parent_id'):
                result.score *= hier_config.get('parent_weight_boost', 1.05)
            
            # Apply sibling context boost
            if hier_config.get('include_sibling_context') and meta.get('children_ids'):
                result.score *= hier_config.get('sibling_weight_boost', 1.02)
            
            # Cap score at 1.0
            result.score = min(result.score, 1.0)
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def _rerank_results(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Apply cross-encoder reranking"""
        if not results or not self._reranker:
            return results
        
        # Prepare pairs for reranking
        pairs = [(query, r.content[:512]) for r in results]
        
        try:
            # Get reranking scores
            scores = self._reranker.predict(pairs)
            
            # Apply sigmoid normalization
            scores = 1 / (1 + np.exp(-scores))
            
            # Combine with original scores
            reranker_weight = self.fusion_config.get('reranker_weight', 0.5)
            
            for result, rerank_score in zip(results, scores):
                original_weight = 1 - reranker_weight
                result.score = (
                    original_weight * result.score +
                    reranker_weight * float(rerank_score)
                )
            
            # Sort by new scores
            results.sort(key=lambda x: x.score, reverse=True)
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
        
        return results
    
    def _optimize_context(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Optimize results to fit within context budget"""
        context_budget = self.fusion_config.get('context_budget', 8000)
        
        optimized = []
        total_tokens = 0
        
        for result in results:
            # Estimate tokens (rough approximation)
            result_tokens = len(result.content.split()) * 1.3
            
            if total_tokens + result_tokens <= context_budget:
                optimized.append(result)
                total_tokens += result_tokens
            else:
                # Try to fit truncated version
                remaining = context_budget - total_tokens
                if remaining > 100:
                    words_to_include = int(remaining / 1.3)
                    result.content = ' '.join(result.content.split()[:words_to_include]) + "..."
                    result.metadata['truncated'] = True
                    optimized.append(result)
                break
        
        return optimized
    
    def evaluate_config(
        self,
        queries: List[Tuple[str, List[float]]],
        ground_truth: List[List[str]],
        config_variations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate different configuration variations.
        
        Args:
            queries: List of (query_text, query_embedding) tuples
            ground_truth: List of relevant chunk IDs for each query
            config_variations: List of config overrides to test
            
        Returns:
            Evaluation results for each configuration
        """
        results = []
        
        for config in config_variations:
            # Apply configuration
            original_config = dict(self.fusion_config)
            self.update_config(config)
            
            # Evaluate on all queries
            scores = []
            for (query, embedding), truth in zip(queries, ground_truth):
                retrieved = self.retrieve(query, embedding)
                retrieved_ids = [r.chunk_id for r in retrieved]
                
                # Calculate metrics
                precision = len(set(retrieved_ids) & set(truth)) / len(retrieved_ids) if retrieved_ids else 0
                recall = len(set(retrieved_ids) & set(truth)) / len(truth) if truth else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                scores.append({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
            
            # Average scores
            avg_scores = {
                'precision': np.mean([s['precision'] for s in scores]),
                'recall': np.mean([s['recall'] for s in scores]),
                'f1': np.mean([s['f1'] for s in scores])
            }
            
            results.append({
                'config': config,
                'scores': avg_scores,
                'detailed_scores': scores
            })
            
            # Restore original config
            self.fusion_config = original_config
        
        # Sort by F1 score
        results.sort(key=lambda x: x['scores']['f1'], reverse=True)
        
        return results