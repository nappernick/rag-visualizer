"""
Query API endpoints for RAG system queries
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime

from ..db import get_session
from ..core.retrieval.hybrid_search import EnhancedHybridSearch
from ..core.temporal.temporal_utils import apply_temporal_boost
from ..services.vector_service import vector_service
from ..services.graph_service import graph_service
from ..services.embedding_service import embedding_service
from ..services.storage import storage_service

router = APIRouter(prefix="/api", tags=["query"])


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
    score: float
    metadata: Dict = {}
    source: str
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    results: List[QueryResult]
    total_results: int
    retrieval_strategy: str
    processing_time_ms: float
    metadata: Dict = {}


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, db: Session = Depends(get_session)):
    """Execute a query against the RAG system."""
    
    start_time = datetime.now()
    
    try:
        results = []
        
        # Get query temporal weight
        temporal_weight = apply_temporal_boost(request.query)
        
        if request.retrieval_strategy in ["vector", "hybrid"]:
            # Generate query embedding
            query_embedding = await embedding_service.generate_embedding(request.query)
            
            if query_embedding:
                # Search similar chunks in Qdrant
                vector_results = await vector_service.search_similar(
                    query_embedding=query_embedding,
                    limit=request.max_results,
                    score_threshold=0.3
                )
                
                # Convert vector results to QueryResult format
                for i, result in enumerate(vector_results):
                    query_result = QueryResult(
                        id=f"vector_{i}",
                        content=result["content"],
                        score=result["score"] * temporal_weight,  # Apply temporal boost
                        metadata={
                            "chunk_type": result["chunk_type"],
                            "temporal_score": result["temporal_score"],
                            "doc_type": result["metadata"].get("doc_type", "default")
                        },
                        source="vector",
                        document_id=result["document_id"],
                        chunk_id=result["chunk_id"]
                    )
                    results.append(query_result)
        
        if request.retrieval_strategy in ["graph", "hybrid"]:
            # Search entities in Neo4j
            graph_entities = await graph_service.search_entities(
                query=request.query,
                limit=min(5, request.max_results)
            )
            
            # Add entity-based results
            for i, entity in enumerate(graph_entities):
                query_result = QueryResult(
                    id=f"graph_{i}",
                    content=f"Entity: {entity['name']} (Type: {entity['entity_type']}, Frequency: {entity['frequency']})",
                    score=min(1.0, entity["frequency"] / 10.0) * temporal_weight,
                    metadata={
                        "entity_id": entity["id"],
                        "entity_type": entity["entity_type"],
                        "frequency": entity["frequency"]
                    },
                    source="graph",
                    document_id=None,
                    chunk_id=None
                )
                results.append(query_result)
        
        # Sort results by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Limit results
        results = results[:request.max_results]
        
        # Apply reranking if requested
        if request.rerank and len(results) > 1:
            # Simple reranking by boosting exact matches
            query_lower = request.query.lower()
            for result in results:
                if any(word in result.content.lower() for word in query_lower.split()):
                    result.score *= 1.2
        
        # Re-sort after reranking
        results.sort(key=lambda x: x.score, reverse=True)
        
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
                "reranked": request.rerank,
                "preset_used": request.preset,
                "vector_results": len([r for r in results if r.source == "vector"]),
                "graph_results": len([r for r in results if r.source == "graph"])
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear-all")
async def clear_all_data(db: Session = Depends(get_session)):
    """Clear all data from the system."""
    
    try:
        results = {"status": "success", "cleared": {}}
        
        # Clear Supabase data
        storage_result = await storage_service.clear_all()
        if storage_result.get("status") == "success":
            results["cleared"]["documents"] = storage_result.get("documents_deleted", 0)
            results["cleared"]["chunks"] = storage_result.get("chunks_deleted", 0)
        
        # Clear Qdrant vectors
        vector_cleared = await vector_service.clear_all()
        results["cleared"]["vectors"] = "cleared" if vector_cleared else "failed"
        
        # Clear Neo4j graph
        graph_result = await graph_service.clear_all()
        if graph_result.get("status") == "success":
            results["cleared"]["entities"] = graph_result.get("nodes_deleted", 0)
            results["cleared"]["relationships"] = graph_result.get("relationships_deleted", 0)
        
        results["message"] = "All data cleared from Supabase, Qdrant, and Neo4j"
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")