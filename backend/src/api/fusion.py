"""
API routes for fusion configuration and tuning
"""
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.retrieval.fusion_controller import FusionController
from ..core.embedding.embedder import Embedder

router = APIRouter(prefix="/api/fusion", tags=["fusion"])

# Initialize fusion controller
fusion_controller = FusionController()
embedder = Embedder({})  # Initialize with default config


class FusionParams(BaseModel):
    """Parameters for fusion configuration"""
    vector_weight: Optional[float] = Field(None, ge=0, le=1)
    graph_weight: Optional[float] = Field(None, ge=0, le=1)
    vector_top_k: Optional[int] = Field(None, ge=1, le=100)
    graph_top_k: Optional[int] = Field(None, ge=1, le=100)
    final_top_k: Optional[int] = Field(None, ge=1, le=50)
    chunk_size: Optional[int] = Field(None, ge=100, le=2000)
    chunk_overlap: Optional[int] = Field(None, ge=0, le=500)
    graph_confidence_threshold: Optional[float] = Field(None, ge=0, le=1)
    graph_expansion_depth: Optional[int] = Field(None, ge=1, le=5)
    entity_relevance_threshold: Optional[float] = Field(None, ge=0, le=1)
    use_reranker: Optional[bool] = None
    reranker_weight: Optional[float] = Field(None, ge=0, le=1)
    context_budget: Optional[int] = Field(None, ge=1000, le=20000)
    prioritize_summaries: Optional[bool] = None
    summary_boost: Optional[float] = Field(None, ge=1, le=2)
    auto_strategy: Optional[bool] = None
    force_hybrid_threshold: Optional[float] = Field(None, ge=0, le=1)


class EvaluationRequest(BaseModel):
    """Request for evaluating fusion configuration"""
    query: str
    ground_truth: List[str]
    configs: Optional[List[Dict[str, Any]]] = None


class QueryWithFusion(BaseModel):
    """Query request with fusion configuration"""
    query: str
    max_results: int = 10
    fusion_config: Optional[Dict[str, Any]] = None
    preset: Optional[str] = None


@router.get("/config")
async def get_fusion_config():
    """Get current fusion configuration"""
    return {
        "status": "ok",
        "config": fusion_controller.fusion_config,
        "presets": fusion_controller.presets
    }


@router.post("/tune")
async def tune_fusion(params: FusionParams):
    """Update fusion configuration dynamically"""
    try:
        # Convert params to dict, excluding None values
        config_updates = {k: v for k, v in params.dict().items() if v is not None}
        
        # Ensure vector and graph weights sum to 1
        if 'vector_weight' in config_updates and 'graph_weight' not in config_updates:
            config_updates['graph_weight'] = 1 - config_updates['vector_weight']
        elif 'graph_weight' in config_updates and 'vector_weight' not in config_updates:
            config_updates['vector_weight'] = 1 - config_updates['graph_weight']
        
        fusion_controller.update_config(config_updates)
        
        return {
            "status": "updated",
            "config": config_updates,
            "full_config": fusion_controller.fusion_config
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/preset/{preset_name}")
async def apply_preset(preset_name: str):
    """Apply a predefined fusion preset"""
    try:
        fusion_controller.apply_preset(preset_name)
        return {
            "status": "applied",
            "preset": preset_name,
            "config": fusion_controller.fusion_config
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to apply preset: {str(e)}")


@router.post("/evaluate")
async def evaluate_fusion(request: EvaluationRequest):
    """Evaluate different fusion configurations"""
    try:
        # Generate embeddings for the query
        query_embedding = embedder.embed_text(request.query)
        
        # Default configurations to test if not provided
        if not request.configs:
            request.configs = [
                {"vector_weight": 0.9, "graph_weight": 0.1},
                {"vector_weight": 0.7, "graph_weight": 0.3},
                {"vector_weight": 0.5, "graph_weight": 0.5},
                {"vector_weight": 0.3, "graph_weight": 0.7},
                {"vector_weight": 0.1, "graph_weight": 0.9},
            ]
        
        # Evaluate configurations
        results = fusion_controller.evaluate_config(
            [(request.query, query_embedding)],
            [request.ground_truth],
            request.configs
        )
        
        return {
            "status": "completed",
            "best": results[0] if results else None,
            "all_results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/query")
async def query_with_fusion(request: QueryWithFusion):
    """Execute query with custom fusion configuration"""
    try:
        # Generate query embedding
        query_embedding = embedder.embed_text(request.query)
        
        # Apply preset if specified
        preset = request.preset
        
        # Apply custom config if provided
        config_overrides = request.fusion_config or {}
        
        # Perform retrieval
        results = fusion_controller.retrieve(
            query=request.query,
            query_embedding=query_embedding,
            preset=preset,
            **config_overrides
        )
        
        # Convert results to response format
        return {
            "query": request.query,
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "content": r.content,
                    "score": r.score,
                    "source": r.source,
                    "metadata": r.metadata
                }
                for r in results[:request.max_results]
            ],
            "total_results": len(results),
            "fusion_strategy": "auto" if fusion_controller.fusion_config.get('auto_strategy') else "manual",
            "active_config": {
                "vector_weight": fusion_controller.fusion_config.get('vector_weight'),
                "graph_weight": fusion_controller.fusion_config.get('graph_weight'),
                "use_reranker": fusion_controller.fusion_config.get('use_reranker')
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/metrics")
async def get_fusion_metrics():
    """Get metrics and statistics about fusion performance"""
    return {
        "status": "ok",
        "metrics": {
            "avg_retrieval_time_ms": 0,  # Would be tracked in production
            "cache_hit_rate": 0,
            "reranker_usage_rate": fusion_controller.fusion_config.get('use_reranker', False),
            "strategy_distribution": {
                "vector": 0,
                "graph": 0,
                "hybrid": 0
            }
        }
    }


@router.post("/optimize")
async def optimize_fusion(
    queries: List[str],
    ground_truths: List[List[str]],
    optimization_metric: str = "f1"
):
    """Automatically optimize fusion configuration based on evaluation data"""
    try:
        # Generate embeddings for all queries
        query_embeddings = [embedder.embed_text(q) for q in queries]
        
        # Grid of configurations to test
        param_grid = []
        for vw in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for rerank in [True, False]:
                for depth in [1, 2, 3]:
                    param_grid.append({
                        "vector_weight": vw,
                        "graph_weight": 1 - vw,
                        "use_reranker": rerank,
                        "graph_expansion_depth": depth
                    })
        
        # Evaluate all configurations
        results = fusion_controller.evaluate_config(
            list(zip(queries, query_embeddings)),
            ground_truths,
            param_grid
        )
        
        # Find best configuration
        best_config = results[0] if results else None
        
        if best_config:
            # Apply the best configuration
            fusion_controller.update_config(best_config['config'])
        
        return {
            "status": "optimized",
            "best_config": best_config,
            "improvement": best_config['scores'][optimization_metric] if best_config else 0,
            "tested_configs": len(param_grid)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")