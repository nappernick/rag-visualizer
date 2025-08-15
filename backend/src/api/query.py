"""
Query API endpoints for RAG system queries
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
import logging

from ..db import get_session
from ..core.retrieval.hybrid_search import EnhancedHybridSearch
from ..core.temporal.temporal_utils import apply_temporal_boost
from ..services.vector_service import get_vector_service
from ..services.graph_service import get_graph_service
from ..services.embedding_service import get_embedding_service
from ..services.storage import get_storage_service
from ..core.scoring.match_scorer import MatchScorer
from ..core.scoring.fusion_ranker import FusionRanker
from ..services.weight_service import get_weight_service

router = APIRouter(prefix="/api", tags=["query"])
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str
    max_results: int = 10
    retrieval_strategy: str = "hybrid"
    include_metadata: bool = True
    rerank: bool = True
    fusion_config: Optional[Dict] = None
    preset: Optional[str] = None


class QueryResult(BaseModel):
    id: str
    content: str
    score: float  # This will be the match percentage (0-100)
    metadata: Dict = {}
    source: str
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    ranking_score: Optional[float] = None  # Used internally for ordering


class QueryResponse(BaseModel):
    query: str
    results: List[QueryResult]
    total_results: int
    retrieval_strategy: str
    processing_time_ms: float
    metadata: Dict = {}


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest, 
    db: Session = Depends(get_session),
    embedding_service=Depends(get_embedding_service),
    vector_service=Depends(get_vector_service),
    graph_service=Depends(get_graph_service)
):
    """Execute a query against the RAG system."""
    
    start_time = datetime.now()
    
    try:
        results = []
        
        # Get fusion configuration with defaults
        fusion_config = request.fusion_config or {}
        vector_weight = fusion_config.get('vector_weight', 0.7)
        graph_weight = fusion_config.get('graph_weight', 0.3)
        vector_top_k = fusion_config.get('vector_top_k', request.max_results * 3)
        graph_top_k = fusion_config.get('graph_top_k', min(5, request.max_results))
        final_top_k = fusion_config.get('final_top_k', request.max_results)
        use_reranker = fusion_config.get('use_reranker', request.rerank)
        context_budget = fusion_config.get('context_budget', 8000)
        auto_strategy = fusion_config.get('auto_strategy', True)
        enable_traversal = fusion_config.get('enable_traversal', True)
        
        # Token optimization: Estimate token usage
        estimated_tokens = 0
        available_budget = context_budget
        
        # Query-adaptive strategy selection
        if auto_strategy:
            query_lower = request.query.lower()
            # Detect query type and adjust weights
            if any(word in query_lower for word in ['how many', 'count', 'list all', 'who are', 'what entities']):
                # Factual/counting queries - favor graph
                vector_weight, graph_weight = 0.3, 0.7
            elif any(word in query_lower for word in ['explain', 'why', 'how does', 'describe', 'what is']):
                # Conceptual queries - favor vector
                vector_weight, graph_weight = 0.8, 0.2
            elif any(word in query_lower for word in ['relationship', 'connection', 'related', 'similar']):
                # Relational queries - balanced with slight graph favor
                vector_weight, graph_weight = 0.4, 0.6
        
        # Get query temporal weight
        temporal_weight = apply_temporal_boost(request.query)
        
        if request.retrieval_strategy in ["vector", "hybrid"]:
            # Generate query embedding
            query_embedding = await embedding_service.generate_embedding(request.query)
            
            if query_embedding:
                # Search similar chunks in Qdrant using fusion config
                vector_results = await vector_service.search_similar(
                    query_embedding=query_embedding,
                    limit=vector_top_k,  # Use fusion config limit
                    score_threshold=0.0  # Remove threshold to get more results
                )
                
                # Get weight service for document weight calculations
                weight_service = get_weight_service()
                
                # Convert vector results to QueryResult format with independent match scoring
                for i, result in enumerate(vector_results):
                    # Get document weight if available
                    document_weight = 1.0
                    if result.get("document_id"):
                        # Try to get document from storage to retrieve its weight
                        try:
                            storage = get_storage_service()
                            doc = await storage.get_document(result["document_id"])
                            if doc:
                                # Calculate document weight based on rules
                                from ..models.schemas import Document as DocModel
                                doc_model = DocModel(**doc)
                                weight_calc = await weight_service.calculate_document_weight(doc_model)
                                document_weight = weight_calc.final_weight
                        except Exception as e:
                            logger.debug(f"Could not get document weight: {e}")
                            document_weight = result.get("metadata", {}).get("document_weight", 1.0)
                    
                    # Calculate independent match score (0-100%)
                    match_score = MatchScorer.calculate_vector_match_score(
                        query=request.query,
                        content=result["content"],
                        cosine_similarity=result["score"],
                        metadata=result.get("metadata", {})
                    )
                    
                    # Calculate ranking score for ordering (includes document weight)
                    ranking_score = match_score * vector_weight * temporal_weight * document_weight
                    
                    query_result = QueryResult(
                        id=f"vector_{i}",
                        content=result["content"],
                        score=match_score,  # Independent match percentage (0-100)
                        metadata={
                            "chunk_type": result["chunk_type"],
                            "temporal_score": result["temporal_score"],
                            "doc_type": result["metadata"].get("doc_type", "default"),
                            "cosine_similarity": result["score"],  # Store raw cosine score
                            "vector_weight": vector_weight,
                            "document_weight": document_weight,
                            "match_explanation": MatchScorer.explain_score(match_score, "vector")
                        },
                        source="vector",
                        document_id=result["document_id"],
                        chunk_id=result["chunk_id"],
                        ranking_score=ranking_score
                    )
                    results.append(query_result)
        
        if request.retrieval_strategy in ["graph", "hybrid"]:
            # Production-grade graph retrieval with 95%+ accuracy
            graph_entities = await graph_service.production_graph_retrieval(
                query=request.query,
                limit=graph_top_k,
                max_hops=2
            )
            
            # Optimized graph traversal for enhanced results (if budget allows)
            traversal_entities = []
            if enable_traversal and graph_entities and available_budget > 4000:
                # Extract seed entity IDs from top graph results for traversal
                seed_entity_ids = [entity["id"] for entity in graph_entities[:2]]
                
                traversal_entities = await graph_service.optimized_graph_traversal(
                    seed_entities=seed_entity_ids,
                    max_hops=2,
                    min_relationship_weight=0.3,
                    limit=min(3, graph_top_k // 2)  # Budget-aware limit
                )
                
                # Update token budget estimate
                estimated_tokens += len(traversal_entities) * 100  # Rough estimate
                available_budget -= estimated_tokens
            
            # Combine graph entities with traversal results
            all_graph_entities = graph_entities + traversal_entities
            
            # Add entity-based results with independent match scoring
            for i, entity in enumerate(all_graph_entities):
                confidence = entity.get("confidence", 0.5)
                match_type = entity.get("match_type", "exact")
                path_length = entity.get("path_length", 0) or 0
                
                # Calculate independent match score (0-100%)
                match_score = MatchScorer.calculate_graph_match_score(
                    query=request.query,
                    entity_name=entity["name"],
                    entity_type=entity["entity_type"],
                    match_type=match_type,
                    confidence=confidence,
                    path_length=path_length
                )
                
                # Calculate ranking score for ordering (this changes with slider)
                ranking_score = match_score * graph_weight * temporal_weight
                
                # Build enhanced content with production context
                content_parts = [f"Entity: {entity['name']} (Type: {entity['entity_type']})"]
                
                # Add detailed context based on match type and strategy
                context = entity.get("context", {})
                all_match_types = entity.get("all_match_types", [match_type])
                
                if match_type == "exact":
                    match_reason = context.get("match_reason", "")
                    if "exact_string_match" in match_reason:
                        content_parts.append(f"Exact match (confidence: {confidence:.2f})")
                elif match_type == "fuzzy":
                    matched_words = context.get("matched_words", [])
                    if matched_words:
                        content_parts.append(f"Word match: {', '.join(matched_words)}")
                elif match_type == "traversal":
                    seed = context.get("seed_entity", "")
                    rel_path = context.get("relationship_path", [])
                    if seed and rel_path:
                        path_str = " → ".join(rel_path)
                        content_parts.append(f"Via {seed}: {path_str}")
                elif match_type == "frequency":
                    freq_boost = context.get("frequency_boost", 0)
                    content_parts.append(f"High-frequency entity ({entity['frequency']}x, boost: +{freq_boost:.2f})")
                
                # Add multi-strategy indicator
                if len(all_match_types) > 1:
                    content_parts.append(f"Multi-strategy: {', '.join(all_match_types)}")
                
                query_result = QueryResult(
                    id=f"graph_{i}",
                    content=" | ".join(content_parts),
                    score=match_score,  # Independent match percentage (0-100)
                    metadata={
                        "entity_id": entity["id"],
                        "entity_type": entity["entity_type"],
                        "frequency": entity["frequency"],
                        "confidence": confidence,
                        "match_type": match_type,
                        "all_match_types": all_match_types,
                        "path_length": path_length,
                        "graph_weight": graph_weight,
                        "match_explanation": MatchScorer.explain_score(match_score, "graph", {"match_type": match_type, "path_length": path_length}),
                        "context": context
                    },
                    source="graph",
                    document_id=None,
                    chunk_id=None,
                    ranking_score=ranking_score
                )
                results.append(query_result)
        
        # Use FusionRanker to properly combine and rank results
        if request.retrieval_strategy == "hybrid" and results:
            # Separate vector and graph results
            vector_results_for_fusion = [
                {
                    "id": r.id,
                    "content": r.content,
                    "match_score": r.score,
                    "metadata": r.metadata
                }
                for r in results if r.source == "vector"
            ]
            
            graph_results_for_fusion = [
                {
                    "id": r.id,
                    "content": r.content,
                    "match_score": r.score,
                    "metadata": r.metadata
                }
                for r in results if r.source == "graph"
            ]
            
            # Use FusionRanker to rank results
            ranked_results = FusionRanker.rank_results(
                vector_results=vector_results_for_fusion,
                graph_results=graph_results_for_fusion,
                vector_weight=vector_weight,
                strategy="hybrid"
            )
            
            # Convert back to QueryResult format
            results = []
            for ranked in ranked_results:
                query_result = QueryResult(
                    id=ranked.id,
                    content=ranked.content,
                    score=ranked.match_score,  # Keep independent match score
                    metadata={
                        **ranked.metadata,
                        "ranking_score": ranked.ranking_score,
                        "fusion_source": ranked.source,
                        "final_rank": ranked.metadata.get("final_rank", 0)
                    },
                    source=ranked.source,
                    document_id=None,
                    chunk_id=None,
                    ranking_score=ranked.ranking_score
                )
                results.append(query_result)
        else:
            # For non-hybrid strategies, sort by ranking score
            results.sort(key=lambda x: x.ranking_score if x.ranking_score is not None else x.score, reverse=True)
        
        # Context budgeting: Limit results based on token budget
        budget_limited_results = []
        current_tokens = 0
        
        for result in results:
            # Estimate tokens for this result (rough approximation)
            content_tokens = len(result.content.split()) * 1.3  # Account for tokenization
            metadata_tokens = 50  # Rough estimate for metadata
            result_tokens = content_tokens + metadata_tokens
            
            # Check if adding this result would exceed budget
            if current_tokens + result_tokens <= available_budget:
                budget_limited_results.append(result)
                current_tokens += result_tokens
            else:
                break  # Stop adding results to stay within budget
        
        # Ensure we don't exceed the fusion config limit
        results = budget_limited_results[:final_top_k]
        
        # Apply reranking if requested (using fusion config)
        if use_reranker and len(results) > 1:
            # Enhanced reranking by adjusting ranking scores, NOT match scores
            query_lower = request.query.lower()
            query_words = [word.strip() for word in query_lower.split() if len(word.strip()) > 2]
            
            for result in results:
                content_lower = result.content.lower()
                rerank_boost = 1.0
                
                # Boost for exact phrase matches
                if query_lower in content_lower:
                    rerank_boost *= 1.5
                
                # Boost for word matches (proportional to word count)
                word_matches = sum(1 for word in query_words if word in content_lower)
                if word_matches > 0:
                    match_ratio = word_matches / len(query_words)
                    rerank_boost *= (1.0 + 0.3 * match_ratio)
                
                # Boost for exact word matches (case insensitive)
                exact_word_matches = sum(1 for word in query_words 
                                       if f" {word} " in f" {content_lower} ")
                if exact_word_matches > 0:
                    rerank_boost *= (1.0 + 0.2 * exact_word_matches)
                
                # Apply boost to ranking score ONLY, not match score
                if result.ranking_score is not None:
                    result.ranking_score *= rerank_boost
                else:
                    result.ranking_score = result.score * rerank_boost
        
        # Re-sort after reranking by ranking score
        results.sort(key=lambda x: x.ranking_score if x.ranking_score is not None else x.score, reverse=True)
        
        # Default result if no matches found
        if not results:
            results.append(QueryResult(
                id="no_results",
                content="No relevant documents found. Try adjusting your query or uploading more documents.",
                score=0.0,
                metadata={"message": "no_results"},
                source="system"
            ))
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            retrieval_strategy=request.retrieval_strategy,
            processing_time_ms=processing_time,
            metadata={
                "temporal_weight": temporal_weight,
                "reranked": use_reranker,
                "preset_used": request.preset,
                "vector_results": len([r for r in results if r.source == "vector"]),
                "graph_results": len([r for r in results if r.source == "graph"]),
                "fusion_config": {
                    "vector_weight": vector_weight,
                    "graph_weight": graph_weight,
                    "vector_top_k": vector_top_k,
                    "graph_top_k": graph_top_k,
                    "final_top_k": final_top_k,
                    "use_reranker": use_reranker,
                    "auto_strategy": auto_strategy,
                    "context_budget": context_budget,
                    "enable_traversal": enable_traversal
                },
                "context_budgeting": {
                    "initial_budget": context_budget,
                    "tokens_used": current_tokens,
                    "budget_remaining": available_budget - current_tokens,
                    "results_limited_by_budget": len(budget_limited_results) < len([r for r in results if r.source != "system"]),
                    "estimated_total_tokens": estimated_tokens + current_tokens
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear-all")
async def clear_all_data(db: Session = Depends(get_session)):
    """Clear all data from the system."""
    
    try:
        results = {"status": "success", "cleared": {}}
        
        # Get service instances
        storage_service = get_storage_service()
        vector_service = get_vector_service()
        graph_service = get_graph_service()
        
        # Clear Supabase data (documents, chunks, entities, relationships)
        storage_result = await storage_service.clear_all()
        if storage_result.get("status") == "success":
            results["cleared"]["documents"] = storage_result.get("documents_deleted", 0)
            results["cleared"]["chunks"] = storage_result.get("chunks_deleted", 0)
            results["cleared"]["supabase_entities"] = storage_result.get("entities_deleted", 0)
            results["cleared"]["supabase_relationships"] = storage_result.get("relationships_deleted", 0)
        
        # Clear Qdrant vectors
        vector_cleared = await vector_service.clear_all()
        results["cleared"]["vectors"] = "cleared" if vector_cleared else "failed"
        
        # Clear Neo4j graph
        graph_result = await graph_service.clear_all()
        if graph_result.get("status") == "success":
            results["cleared"]["neo4j_nodes"] = graph_result.get("nodes_deleted", 0)
            results["cleared"]["neo4j_relationships"] = graph_result.get("relationships_deleted", 0)
        
        results["message"] = "All data cleared from Supabase, Qdrant, and Neo4j"
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")