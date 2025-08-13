"""
Visualization API endpoints for document visualization data
"""
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..db import get_session
from ..models import Document, Chunk, Entity, Relationship

router = APIRouter(prefix="/api/visualization", tags=["visualization"])


class ChunkNode(BaseModel):
    id: str
    content: str
    position: int
    metadata: Dict = {}
    embedding_position: Optional[List[float]] = None


class EntityNode(BaseModel):
    id: str
    name: str
    type: str
    metadata: Dict = {}


class Edge(BaseModel):
    source: str
    target: str
    type: str
    weight: float = 1.0
    metadata: Dict = {}


class VisualizationData(BaseModel):
    document_id: str
    title: str
    chunks: List[ChunkNode]
    entities: List[EntityNode]
    relationships: List[Edge]
    stats: Dict


@router.get("/{document_id}", response_model=VisualizationData)
async def get_visualization_data(document_id: str, db: Session = Depends(get_session)):
    """Get complete visualization data for a document."""
    
    # Mock implementation - in production would fetch from database
    return VisualizationData(
        document_id=document_id,
        title=f"Document {document_id}",
        chunks=[
            ChunkNode(
                id=f"chunk_{document_id}_1",
                content="This is a sample chunk for visualization.",
                position=0,
                metadata={"length": 40}
            ),
            ChunkNode(
                id=f"chunk_{document_id}_2",
                content="Another chunk with different content.",
                position=1,
                metadata={"length": 37}
            )
        ],
        entities=[
            EntityNode(
                id=f"entity_{document_id}_1",
                name="Sample Entity",
                type="Concept",
                metadata={"importance": 0.8}
            ),
            EntityNode(
                id=f"entity_{document_id}_2",
                name="Another Entity",
                type="Person",
                metadata={"importance": 0.6}
            )
        ],
        relationships=[
            Edge(
                source=f"chunk_{document_id}_1",
                target=f"entity_{document_id}_1",
                type="mentions",
                weight=0.9
            ),
            Edge(
                source=f"entity_{document_id}_1",
                target=f"entity_{document_id}_2",
                type="related_to",
                weight=0.7
            )
        ],
        stats={
            "total_chunks": 2,
            "total_entities": 2,
            "total_relationships": 2,
            "avg_chunk_size": 38.5
        }
    )