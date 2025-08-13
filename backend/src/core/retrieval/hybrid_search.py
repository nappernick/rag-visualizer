"""
Enhanced Hybrid Search with Reciprocal Rank Fusion (RRF) and multi-strategy retrieval.
Implements parallel vector, keyword, and metadata-based retrieval with intelligent merging.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import redis
from redis.commands.search import Search
from redis.commands.search.query import Query
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ...models import RetrievalResult, Document, Chunk
from .vector_retriever import VectorRetriever
from ...db import get_session
from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for hybrid search"""
    vector_weight: float = 0.4
    keyword_weight: float = 0.3
    metadata_weight: float = 0.3
    vector_top_k: int = 20
    keyword_top_k: int = 20
    metadata_top_k: int = 10
    rrf_k: int = 60  # RRF constant
    use_parallel: bool = True
    keyword_boost_terms: List[str] = None
    metadata_filters: Dict[str, Any] = None


class RedisSearchIndex:
    """Redis-based keyword and metadata search"""
    
    def __init__(self, redis_host: str = None, redis_port: int = None):
        self.enabled = False
        
        # Check if Redis is enabled in configuration
        if not config.is_redis_enabled():
            logger.info("Redis is disabled in configuration")
            self.redis_client = None
            self.search_client = None
            return
        
        # Use provided parameters or fall back to config
        host = redis_host or config.REDIS_HOST
        port = redis_port or config.REDIS_PORT
        
        try:
            self.redis_client = redis.Redis(host=host, port=port, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            self.search_client = self.redis_client.ft('chunks')
            self._ensure_index()
            self.enabled = True
            logger.info(f"Redis search connected to {host}:{port}")
        except Exception as e:
            logger.warning(f"Redis search not available: {e}. Using fallback search.")
            self.redis_client = None
            self.search_client = None
    
    def _ensure_index(self):
        """Ensure Redis search index exists"""
        if not self.redis_client:
            return
        try:
            # Check if index exists
            self.search_client.info()
        except:
            # Create index if it doesn't exist
            logger.info("Creating Redis search index for chunks")
            try:
                from redis.commands.search.field import TextField, NumericField, TagField
                from redis.commands.search.indexDefinition import IndexDefinition, IndexType
            except ImportError:
                # Fallback for different Redis versions
                logger.warning("Redis search module not available")
                self.enabled = False
                return
            
            schema = [
                TextField('content', weight=1.0),
                TextField('document_title', weight=0.5),
                TagField('chunk_type'),
                TagField('document_id'),
                NumericField('position'),
                TagField('tags', separator=','),
            ]
            
            definition = IndexDefinition(
                prefix=['chunk:'],
                index_type=IndexType.HASH
            )
            
            try:
                self.redis_client.ft('chunks').create_index(
                    schema,
                    definition=definition
                )
                logger.info("Redis search index created successfully")
            except Exception as e:
                logger.warning(f"Could not create Redis index: {e}")
    
    def index_chunk(self, chunk: Chunk, document: Document):
        """Index a chunk in Redis for keyword search"""
        if not self.enabled or not self.redis_client:
            return
        
        chunk_key = f"chunk:{chunk.id}"
        
        # Prepare data for indexing
        data = {
            'content': chunk.content,
            'document_title': document.title,
            'chunk_type': chunk.metadata.get('chunk_type', 'standard'),
            'document_id': document.id,
            'position': chunk.position,
            'tags': ','.join(chunk.metadata.get('tags', []))
        }
        
        try:
            # Store in Redis
            self.redis_client.hset(chunk_key, mapping=data)
        except Exception as e:
            logger.warning(f"Could not index chunk in Redis: {e}")
    
    def search_keywords(self, query: str, top_k: int = 20, boost_terms: List[str] = None) -> List[Dict]:
        """Perform keyword search using Redis Search"""
        if not self.enabled or not self.search_client:
            return []
        
        try:
            # Build search query with optional boosting
            search_query = query
            if boost_terms:
                # Boost specific terms if provided
                for term in boost_terms:
                    if term.lower() in query.lower():
                        search_query = search_query.replace(term, f"{term}^2")
            
            # Create Redis query
            q = Query(search_query).paging(0, top_k).with_scores()
            
            # Execute search
            results = self.search_client.search(q)
            
            # Format results
            formatted_results = []
            for doc in results.docs:
                formatted_results.append({
                    'chunk_id': doc.id.replace('chunk:', ''),
                    'score': float(doc.score) if hasattr(doc, 'score') else 1.0,
                    'content': doc.content,
                    'document_id': doc.document_id,
                    'metadata': {
                        'chunk_type': doc.chunk_type,
                        'position': int(doc.position) if doc.position else 0,
                        'tags': doc.tags.split(',') if doc.tags else []
                    }
                })
            
            return formatted_results
            
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            return []
    
    def search_metadata(self, filters: Dict[str, Any], top_k: int = 10) -> List[Dict]:
        """Search based on metadata filters"""
        if not self.enabled or not self.search_client:
            return []
        
        try:
            # Build filter query
            filter_parts = []
            
            if 'document_id' in filters:
                filter_parts.append(f"@document_id:{{{filters['document_id']}}}")
            
            if 'chunk_type' in filters:
                filter_parts.append(f"@chunk_type:{{{filters['chunk_type']}}}")
            
            if 'tags' in filters:
                for tag in filters['tags']:
                    filter_parts.append(f"@tags:{{{tag}}}")
            
            if not filter_parts:
                return []
            
            # Combine filters
            filter_query = ' '.join(filter_parts)
            
            # Create query
            q = Query(filter_query).paging(0, top_k)
            
            # Execute search
            results = self.search_client.search(q)
            
            # Format results
            formatted_results = []
            for doc in results.docs:
                formatted_results.append({
                    'chunk_id': doc.id.replace('chunk:', ''),
                    'score': 1.0,  # Metadata matches are binary
                    'content': doc.content,
                    'document_id': doc.document_id,
                    'metadata': {
                        'chunk_type': doc.chunk_type,
                        'position': int(doc.position) if doc.position else 0,
                        'tags': doc.tags.split(',') if doc.tags else []
                    }
                })
            
            return formatted_results
            
        except Exception as e:
            logger.warning(f"Metadata search failed: {e}")
            return []


class EnhancedHybridSearch:
    """
    Enhanced hybrid search combining semantic, keyword, and metadata-based retrieval
    with Reciprocal Rank Fusion for optimal result merging.
    """
    
    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        
        # Initialize retrievers
        self.vector_retriever = VectorRetriever({
            'vector_store': {
                'provider': 'qdrant',
                'host': 'localhost',
                'port': 6333,
                'collection_name': 'rag_chunks'
            }
        })
        
        self.keyword_index = RedisSearchIndex()
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        logger.info("Enhanced Hybrid Search initialized")
    
    async def search(
        self,
        query: str,
        query_embedding: List[float],
        config_override: Optional[SearchConfig] = None
    ) -> List[RetrievalResult]:
        """
        Perform enhanced hybrid search with multiple strategies.
        
        Args:
            query: Text query
            query_embedding: Query embedding vector
            config_override: Optional config to override defaults
            
        Returns:
            Merged and ranked retrieval results
        """
        config = config_override or self.config
        
        if config.use_parallel:
            # Parallel retrieval for better performance
            results = await self._parallel_retrieve(query, query_embedding, config)
        else:
            # Sequential retrieval
            results = await self._sequential_retrieve(query, query_embedding, config)
        
        # Apply Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            results,
            weights=[config.vector_weight, config.keyword_weight, config.metadata_weight],
            k=config.rrf_k
        )
        
        return fused_results
    
    async def _parallel_retrieve(
        self,
        query: str,
        query_embedding: List[float],
        config: SearchConfig
    ) -> Dict[str, List[RetrievalResult]]:
        """Execute retrieval strategies in parallel"""
        
        # Create async tasks for each retrieval method
        tasks = []
        
        # Vector search task
        tasks.append(
            asyncio.create_task(
                self._async_vector_search(query_embedding, config.vector_top_k)
            )
        )
        
        # Keyword search task
        tasks.append(
            asyncio.create_task(
                self._async_keyword_search(query, config.keyword_top_k, config.keyword_boost_terms)
            )
        )
        
        # Metadata search task (if filters provided)
        if config.metadata_filters:
            tasks.append(
                asyncio.create_task(
                    self._async_metadata_search(config.metadata_filters, config.metadata_top_k)
                )
            )
        else:
            tasks.append(asyncio.create_task(self._async_empty_results()))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        return {
            'vector': results[0],
            'keyword': results[1],
            'metadata': results[2]
        }
    
    async def _sequential_retrieve(
        self,
        query: str,
        query_embedding: List[float],
        config: SearchConfig
    ) -> Dict[str, List[RetrievalResult]]:
        """Execute retrieval strategies sequentially"""
        
        results = {}
        
        # Vector search
        results['vector'] = await self._async_vector_search(query_embedding, config.vector_top_k)
        
        # Keyword search
        results['keyword'] = await self._async_keyword_search(
            query, config.keyword_top_k, config.keyword_boost_terms
        )
        
        # Metadata search
        if config.metadata_filters:
            results['metadata'] = await self._async_metadata_search(
                config.metadata_filters, config.metadata_top_k
            )
        else:
            results['metadata'] = []
        
        return results
    
    async def _async_vector_search(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[RetrievalResult]:
        """Async wrapper for vector search"""
        try:
            results = self.vector_retriever.retrieve(query_embedding, top_k)
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _async_keyword_search(
        self,
        query: str,
        top_k: int,
        boost_terms: List[str] = None
    ) -> List[RetrievalResult]:
        """Async wrapper for keyword search"""
        try:
            # Run keyword search in thread pool
            loop = asyncio.get_event_loop()
            keyword_results = await loop.run_in_executor(
                self.executor,
                self.keyword_index.search_keywords,
                query,
                top_k,
                boost_terms
            )
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for result in keyword_results:
                retrieval_results.append(RetrievalResult(
                    chunk_id=result['chunk_id'],
                    content=result['content'],
                    score=result['score'],
                    metadata=result['metadata'],
                    source='keyword'
                ))
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    async def _async_metadata_search(
        self,
        filters: Dict[str, Any],
        top_k: int
    ) -> List[RetrievalResult]:
        """Async wrapper for metadata search"""
        try:
            # Run metadata search in thread pool
            loop = asyncio.get_event_loop()
            metadata_results = await loop.run_in_executor(
                self.executor,
                self.keyword_index.search_metadata,
                filters,
                top_k
            )
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for result in metadata_results:
                retrieval_results.append(RetrievalResult(
                    chunk_id=result['chunk_id'],
                    content=result['content'],
                    score=result['score'],
                    metadata=result['metadata'],
                    source='metadata'
                ))
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []
    
    async def _async_empty_results(self) -> List[RetrievalResult]:
        """Return empty results for disabled strategies"""
        return []
    
    def _reciprocal_rank_fusion(
        self,
        result_sets: Dict[str, List[RetrievalResult]],
        weights: List[float],
        k: int = 60
    ) -> List[RetrievalResult]:
        """
        Apply Reciprocal Rank Fusion algorithm to merge results.
        
        RRF Score = Σ(weight_i * 1/(k + rank_i))
        
        Args:
            result_sets: Dictionary of result lists from different strategies
            weights: Weights for each strategy [vector, keyword, metadata]
            k: RRF constant (typically 60)
            
        Returns:
            Merged and ranked results
        """
        chunk_scores = {}
        chunk_data = {}
        
        # Process each result set
        strategies = ['vector', 'keyword', 'metadata']
        
        for strategy, weight in zip(strategies, weights):
            results = result_sets.get(strategy, [])
            
            for rank, result in enumerate(results, 1):
                chunk_id = result.chunk_id
                
                # Calculate RRF score for this result
                rrf_score = weight * (1 / (k + rank))
                
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = 0
                    chunk_data[chunk_id] = result
                else:
                    # Update source to show multiple strategies found this chunk
                    existing = chunk_data[chunk_id]
                    if existing.source != 'hybrid' and existing.source != result.source:
                        existing.source = 'hybrid'
                    
                    # Merge metadata
                    existing.metadata.update(result.metadata)
                
                # Add to cumulative RRF score
                chunk_scores[chunk_id] += rrf_score
        
        # Create final ranked list
        merged_results = []
        for chunk_id, score in chunk_scores.items():
            result = chunk_data[chunk_id]
            result.score = score  # Use RRF score
            merged_results.append(result)
        
        # Sort by RRF score
        merged_results.sort(key=lambda x: x.score, reverse=True)
        
        # Log fusion statistics
        logger.info(f"RRF Fusion complete: {len(merged_results)} unique chunks from "
                   f"{sum(len(r) for r in result_sets.values())} total results")
        
        return merged_results
    
    def _normalize_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Normalize scores to [0, 1] range"""
        if not results:
            return results
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same
            for r in results:
                r.score = 0.5
        else:
            # Normalize to [0, 1]
            score_range = max_score - min_score
            for r in results:
                r.score = (r.score - min_score) / score_range
        
        return results
    
    async def index_document_chunks(self, document: Document, chunks: List[Chunk]):
        """Index document chunks for keyword and metadata search"""
        try:
            # Index each chunk in Redis
            for chunk in chunks:
                self.keyword_index.index_chunk(chunk, document)
            
            logger.info(f"Indexed {len(chunks)} chunks for document {document.id}")
            
        except Exception as e:
            logger.error(f"Failed to index chunks: {e}")
    
    def calculate_metrics(
        self,
        retrieved: List[RetrievalResult],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        """Calculate retrieval metrics"""
        retrieved_ids = [r.chunk_id for r in retrieved]
        
        if not retrieved_ids:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Calculate metrics
        true_positives = len(set(retrieved_ids) & set(ground_truth))
        
        precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
        recall = true_positives / len(ground_truth) if ground_truth else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'retrieved_count': len(retrieved_ids),
            'relevant_count': len(ground_truth)
        }