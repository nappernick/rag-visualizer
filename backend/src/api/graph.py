"""
Graph API endpoints for entity extraction and relationship management
"""
from typing import List, Optional, Dict
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
import hashlib

from ..db import get_session
from ..models import Entity, Relationship
from ..services.graph_service import graph_service
from ..services.entity_service import entity_service

router = APIRouter(prefix="/api/graph", tags=["graph"])


class EntityResponse(BaseModel):
    id: str
    name: str
    entity_type: str  # Frontend expects entity_type, not type
    document_ids: List[str] = []
    chunk_ids: List[str] = []
    frequency: int = 1
    metadata: Dict = {}


class RelationshipResponse(BaseModel):
    id: str
    source_entity_id: str  # Frontend expects source_entity_id
    target_entity_id: str  # Frontend expects target_entity_id
    relationship_type: str  # Frontend expects relationship_type
    weight: float = 1.0
    document_ids: List[str] = []
    metadata: Dict = {}


class GraphExtractionRequest(BaseModel):
    document_id: str
    chunks: List[Dict]
    extract_entities: bool = True
    extract_relationships: bool = True
    use_spacy: bool = True
    use_claude: Optional[bool] = True  # Default to Claude


class GraphExtractionResponse(BaseModel):
    entities: List[EntityResponse]
    relationships: List[RelationshipResponse]


@router.post("/extract", response_model=GraphExtractionResponse)
async def extract_graph(request: GraphExtractionRequest, db: Session = Depends(get_session)):
    """Extract entities and relationships from document chunks."""
    
    entities = []
    relationships = []
    
    if request.extract_entities:
        # Combine all chunk content for entity extraction
        full_text = "\n\n".join([chunk.get("content", "") for chunk in request.chunks])
        chunk_ids = [chunk.get("id", "") for chunk in request.chunks if chunk.get("id")]
        
        # Extract entities using the entity service
        extracted_entities = await entity_service.extract_entities(
            text=full_text,
            document_id=request.document_id,
            chunk_ids=chunk_ids,
            use_claude=request.use_claude
        )
        
        # Convert to response format
        for entity_data in extracted_entities:
            entities.append(EntityResponse(
                id=entity_data["id"],
                name=entity_data["name"],
                entity_type=entity_data["entity_type"],
                document_ids=entity_data["document_ids"],
                chunk_ids=entity_data["chunk_ids"],
                frequency=entity_data["frequency"],
                metadata=entity_data["metadata"]
            ))
        
        # Store entities in Neo4j
        if extracted_entities:
            await graph_service.store_entities(extracted_entities)
            await graph_service.link_entities_to_document(request.document_id, [e["id"] for e in extracted_entities])
            
            # Link entities to chunks
            for chunk in request.chunks:
                chunk_id = chunk.get("id")
                if chunk_id:
                    # Find entities mentioned in this chunk
                    chunk_content = chunk.get("content", "").lower()
                    chunk_entity_ids = [
                        e["id"] for e in extracted_entities 
                        if e["name"].lower() in chunk_content
                    ]
                    if chunk_entity_ids:
                        await graph_service.link_entities_to_chunks(chunk_id, chunk_entity_ids)
    
    if request.extract_relationships and entities:
        # Extract relationships using the entity service
        full_text = "\n\n".join([chunk.get("content", "") for chunk in request.chunks])
        entity_dicts = [{
            "id": e.id,
            "name": e.name,
            "entity_type": e.entity_type,
            "frequency": e.frequency
        } for e in entities]
        
        extracted_relationships = await entity_service.extract_relationships(
            entities=entity_dicts,
            text=full_text,
            document_id=request.document_id
        )
        
        # Convert to response format
        for rel_data in extracted_relationships:
            relationships.append(RelationshipResponse(
                id=rel_data["id"],
                source_entity_id=rel_data["source_entity_id"],
                target_entity_id=rel_data["target_entity_id"],
                relationship_type=rel_data["relationship_type"],
                weight=rel_data["weight"],
                document_ids=rel_data["document_ids"],
                metadata=rel_data["metadata"]
            ))
        
        # Store relationships in Neo4j
        if extracted_relationships:
            await graph_service.store_relationships(extracted_relationships)
    
    return GraphExtractionResponse(
        entities=entities,
        relationships=relationships
    )


@router.get("/{document_id}/entities", response_model=List[EntityResponse])
async def get_entities(document_id: str, db: Session = Depends(get_session)):
    """Get all entities for a document."""
    entities = await graph_service.get_document_entities(document_id)
    
    # Convert to response format
    return [
        EntityResponse(
            id=entity["id"],
            name=entity["name"],
            entity_type=entity["entity_type"],
            document_ids=entity["document_ids"],
            chunk_ids=entity["chunk_ids"],
            frequency=entity["frequency"],
            metadata=entity["metadata"]
        )
        for entity in entities
    ]


@router.get("/{document_id}/relationships", response_model=List[RelationshipResponse])
async def get_relationships(document_id: str, db: Session = Depends(get_session)):
    """Get all relationships for a document."""
    relationships = await graph_service.get_document_relationships(document_id)
    
    # Convert to response format
    return [
        RelationshipResponse(
            id=rel["id"],
            source_entity_id=rel["source_entity_id"],
            target_entity_id=rel["target_entity_id"],
            relationship_type=rel["relationship_type"],
            weight=rel["weight"],
            document_ids=rel["document_ids"],
            metadata=rel["metadata"]
        )
        for rel in relationships
    ]


@router.post("/link-documents")
async def link_documents(request: Dict, db: Session = Depends(get_session)):
    """Link documents in the graph."""
    # Mock implementation
    return {"status": "success", "message": "Documents linked successfully"}