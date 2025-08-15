"""
Refactored Demo API - Thin handlers using service layer
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import os
import boto3
from sqlalchemy.orm import Session

from ..database import get_session
from ..services.demo.search_service import DemoSearchService
from ..services.demo.suggestion_service import SuggestionService
from ..services.demo.document_analysis_service import DocumentAnalysisService
from ..services.demo.graph_exploration_service import GraphExplorationService
from ..core.retrieval.fusion_controller import FusionController
from ..services.vector_service import VectorService
from ..services.graph_service import GraphService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/demo", tags=["demo"])

# Initialize services
bedrock_client = None
if os.getenv("AWS_ACCESS_KEY_ID"):
    try:
        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )
        logger.info("Bedrock client initialized for demo API")
    except Exception as e:
        logger.warning(f"Could not initialize Bedrock client: {e}")

# Service instances (would typically use dependency injection)
fusion_controller = FusionController()
vector_service = VectorService()
graph_service = GraphService()

search_service = DemoSearchService(
    fusion_controller=fusion_controller,
    vector_service=vector_service,
    graph_service=graph_service,
    bedrock_client=bedrock_client
)

suggestion_service = SuggestionService(bedrock_client=bedrock_client)


# Request/Response Models
class DemoSearchRequest(BaseModel):
    query: str
    use_smart_routing: bool = True
    include_explanation: bool = True
    max_results: int = 10
    strategy: Optional[str] = None

class DemoSearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    strategy: str
    total_results: int
    search_time: float
    explanation: Optional[str]
    highlights: List[str]
    metadata: Dict[str, Any]

class SuggestionRequest(BaseModel):
    partial_query: str
    context: Optional[List[str]] = None
    max_suggestions: int = 5

class SummarizeRequest(BaseModel):
    document_ids: List[str]
    summary_type: str = "brief"  # brief, detailed, technical
    max_length: int = 500

class ExploreRequest(BaseModel):
    entity_name: Optional[str] = None
    document_id: Optional[str] = None
    exploration_depth: int = 2
    include_relationships: bool = True

class PathFindRequest(BaseModel):
    start_entity: str
    end_entity: str
    max_path_length: int = 5

class AnalyzeRequest(BaseModel):
    document_ids: List[str]
    analysis_type: str = "overview"  # overview, detailed, comparative


# API Endpoints
@router.post("/search", response_model=DemoSearchResponse)
async def enhanced_search(
    request: DemoSearchRequest,
    db: Session = Depends(get_session)
) -> DemoSearchResponse:
    """
    Perform enhanced search with smart routing and explanations
    """
    try:
        result = await search_service.enhanced_search(
            query=request.query,
            use_smart_routing=request.use_smart_routing,
            include_explanation=request.include_explanation,
            max_results=request.max_results,
            strategy=request.strategy
        )
        
        return DemoSearchResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in enhanced search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/suggestions")
async def get_suggestions(
    request: SuggestionRequest,
    db: Session = Depends(get_session)
) -> Dict[str, Any]:
    """
    Get query suggestions based on partial input
    """
    try:
        suggestions = await suggestion_service.get_suggestions(
            partial_query=request.partial_query,
            context=request.context,
            max_suggestions=request.max_suggestions
        )
        
        return {
            "suggestions": suggestions,
            "partial_query": request.partial_query
        }
        
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        # Return simple fallback suggestions
        return {
            "suggestions": suggestion_service.get_simple_suggestions(request.partial_query),
            "partial_query": request.partial_query,
            "fallback": True
        }


@router.post("/query/decompose")
async def decompose_query(
    query: str,
    db: Session = Depends(get_session)
) -> Dict[str, Any]:
    """
    Decompose a complex query into sub-queries
    """
    try:
        # This would use a QueryDecompositionService
        sub_queries = [
            f"What is {query.split()[0]}?",
            f"How does {query.split()[-1]} work?",
            f"Examples of {query}"
        ]
        
        return {
            "original_query": query,
            "sub_queries": sub_queries,
            "strategy": "decomposition"
        }
        
    except Exception as e:
        logger.error(f"Error decomposing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/summarize")
async def summarize_documents(
    request: SummarizeRequest,
    db: Session = Depends(get_session)
) -> Dict[str, Any]:
    """
    Generate summaries for specified documents
    """
    # Placeholder - would use DocumentAnalysisService
    return {
        "summaries": [
            {
                "document_id": doc_id,
                "summary": f"Summary of document {doc_id}",
                "type": request.summary_type
            }
            for doc_id in request.document_ids
        ],
        "total_documents": len(request.document_ids),
        "summary_type": request.summary_type
    }


@router.post("/graph/explore")
async def explore_graph(
    request: ExploreRequest,
    db: Session = Depends(get_session)
) -> Dict[str, Any]:
    """
    Explore the knowledge graph from a starting point
    """
    # Placeholder - would use GraphExplorationService
    return {
        "exploration": {
            "start_entity": request.entity_name,
            "depth": request.exploration_depth,
            "entities_found": 10,
            "relationships_found": 15
        },
        "entities": [],
        "relationships": []
    }


@router.post("/graph/find-path")
async def find_path(
    request: PathFindRequest,
    db: Session = Depends(get_session)
) -> Dict[str, Any]:
    """
    Find connection paths between two entities
    """
    # Placeholder - would use GraphExplorationService
    return {
        "paths": [
            {
                "path": [request.start_entity, "intermediate", request.end_entity],
                "length": 2,
                "confidence": 0.85
            }
        ],
        "start": request.start_entity,
        "end": request.end_entity,
        "paths_found": 1
    }


@router.post("/documents/analyze")
async def analyze_documents(
    request: AnalyzeRequest,
    db: Session = Depends(get_session)
) -> Dict[str, Any]:
    """
    Perform analysis on a set of documents
    """
    # Placeholder - would use DocumentAnalysisService
    return {
        "analysis": {
            "type": request.analysis_type,
            "documents_analyzed": len(request.document_ids),
            "key_findings": [
                "Finding 1",
                "Finding 2",
                "Finding 3"
            ],
            "statistics": {
                "total_entities": 100,
                "total_relationships": 150,
                "avg_chunk_size": 500
            }
        },
        "document_ids": request.document_ids
    }