"""
Graph API endpoints for entity extraction and relationship management
"""
from typing import List, Optional, Dict
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
import logging
import hashlib

from ..db import get_session
from ..models import Entity, Relationship
from ..services.graph_service import get_graph_service
from ..services.entity_service import get_entity_service
from ..services.storage import get_storage_service

logger = logging.getLogger(__name__)

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
        
        # Extract entities and relationships using the entity service
        entity_service = get_entity_service()
        extracted_entities, extracted_relationships = await entity_service.extract_entities(
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
        
        # Store entities in both Neo4j and Supabase
        if extracted_entities:
            # Store in Neo4j
            graph_service = get_graph_service()
            await graph_service.store_entities(extracted_entities)
            await graph_service.link_entities_to_document(request.document_id, [e["id"] for e in extracted_entities])
            
            # Store in Supabase
            storage_service = get_storage_service()
            await storage_service.store_entities(extracted_entities)
            
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
    
    if request.extract_relationships:
        # Use relationships extracted by Claude (already in extracted_relationships)
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
        
        # Store relationships in both Neo4j and Supabase
        if extracted_relationships:
            # Store in Neo4j
            await graph_service.store_relationships(extracted_relationships)
            
            # Store in Supabase
            storage_service = get_storage_service()
            await storage_service.store_relationships(extracted_relationships)
    
    return GraphExtractionResponse(
        entities=entities,
        relationships=relationships
    )


@router.get("/{document_id}/entities", response_model=List[EntityResponse])
async def get_entities(document_id: str, 
                      db: Session = Depends(get_session),
                      graph_service=Depends(get_graph_service),
                      storage_service=Depends(get_storage_service)):
    """Get all entities for a document."""
    # Try to get from Supabase first
    entities = await storage_service.get_entities(document_id)
    
    # Fallback to Neo4j if Supabase is empty
    if not entities:
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
async def get_relationships(document_id: str, 
                           db: Session = Depends(get_session),
                           graph_service=Depends(get_graph_service),
                           storage_service=Depends(get_storage_service)):
    """Get all relationships for a document."""
    # Try to get from Supabase first
    relationships = await storage_service.get_relationships(document_id)
    
    # Fallback to Neo4j if Supabase is empty
    if not relationships:
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
async def link_documents(
    db: Session = Depends(get_session),
    storage_service=Depends(get_storage_service),
    graph_service=Depends(get_graph_service)
):
    """Link documents in the graph by finding matching entities."""
    try:
        # Get all entities from Supabase
        all_entities = []
        documents = await storage_service.get_documents()
        
        for doc in documents:
            entities = await storage_service.get_entities(doc.get("id"))
            all_entities.extend(entities)
        
        # Enhanced entity matching with multiple strategies
        entity_groups_exact = {}
        entity_groups_partial = {}
        entity_groups_semantic = {}
        
        # Strategy 1: Exact name matching (existing)
        for entity in all_entities:
            name = entity.get("name", "").lower().strip()
            if name and len(name) > 2:  # Minimum length for meaningful matching
                if name not in entity_groups_exact:
                    entity_groups_exact[name] = []
                entity_groups_exact[name].append(entity)
        
        # Strategy 2: Partial/fuzzy matching for similar names
        import difflib
        for entity in all_entities:
            name = entity.get("name", "").lower().strip()
            if name and len(name) > 3:
                # Find similar names using fuzzy matching
                for existing_name in entity_groups_partial.keys():
                    similarity = difflib.SequenceMatcher(None, name, existing_name).ratio()
                    if similarity > 0.8 and name != existing_name:  # 80% similarity threshold
                        if existing_name not in entity_groups_partial:
                            entity_groups_partial[existing_name] = []
                        entity_groups_partial[existing_name].append(entity)
                        break
                else:
                    # No similar name found, create new group
                    if name not in entity_groups_partial:
                        entity_groups_partial[name] = []
                    entity_groups_partial[name].append(entity)
        
        # Strategy 3: Entity type clustering (entities of same type that might be related)
        type_groups = {}
        for entity in all_entities:
            entity_type = entity.get("entity_type", "").lower().strip()
            name = entity.get("name", "").lower().strip()
            if entity_type and len(entity_type) > 2 and len(name) > 2:
                key = f"{entity_type}_{name[:3]}"  # Type + first 3 chars
                if key not in type_groups:
                    type_groups[key] = []
                type_groups[key].append(entity)
        
        cross_relationships = []
        entity_matches = 0
        
        # Process exact matches (highest confidence)
        for name, entities in entity_groups_exact.items():
            if len(entities) > 1:
                doc_ids = list(set([e.get("document_ids", [None])[0] for e in entities if e.get("document_ids")]))
                if len(doc_ids) > 1:
                    entity_matches += 1
                    for i, entity1 in enumerate(entities):
                        for entity2 in entities[i+1:]:
                            relationship = {
                                "id": f"exact_{entity1.get('id')}_{entity2.get('id')}",
                                "source_entity_id": entity1.get("id"),
                                "target_entity_id": entity2.get("id"),
                                "relationship_type": "CROSS_DOCUMENT_EXACT",
                                "weight": 0.95,
                                "document_ids": doc_ids,
                                "metadata": {
                                    "created_by": "link_documents",
                                    "match_type": "name_exact",
                                    "entity_name": name,
                                    "confidence": 0.95
                                }
                            }
                            cross_relationships.append(relationship)
        
        # Process fuzzy matches (medium confidence)
        for name, entities in entity_groups_partial.items():
            if len(entities) > 1:
                doc_ids = list(set([e.get("document_ids", [None])[0] for e in entities if e.get("document_ids")]))
                if len(doc_ids) > 1:
                    entity_matches += 1
                    for i, entity1 in enumerate(entities):
                        for entity2 in entities[i+1:]:
                            # Calculate actual similarity
                            sim = difflib.SequenceMatcher(None, 
                                entity1.get("name", "").lower(), 
                                entity2.get("name", "").lower()).ratio()
                            if sim > 0.8:
                                relationship = {
                                    "id": f"fuzzy_{entity1.get('id')}_{entity2.get('id')}",
                                    "source_entity_id": entity1.get("id"),
                                    "target_entity_id": entity2.get("id"),
                                    "relationship_type": "CROSS_DOCUMENT_SIMILAR",
                                    "weight": sim * 0.8,  # Scale by similarity
                                    "document_ids": doc_ids,
                                    "metadata": {
                                        "created_by": "link_documents",
                                        "match_type": "name_fuzzy",
                                        "similarity_score": sim,
                                        "confidence": sim * 0.8
                                    }
                                }
                                cross_relationships.append(relationship)
        
        # Process type-based clustering (lower confidence)
        for type_key, entities in type_groups.items():
            if len(entities) > 1:
                doc_ids = list(set([e.get("document_ids", [None])[0] for e in entities if e.get("document_ids")]))
                if len(doc_ids) > 1:
                    # Only create type-based relationships if entities have similar contexts
                    entity_type = type_key.split('_')[0]
                    for i, entity1 in enumerate(entities):
                        for entity2 in entities[i+1:]:
                            # Check if they're not already connected by exact or fuzzy matches
                            exact_exists = any(r.get("source_entity_id") == entity1.get("id") and 
                                             r.get("target_entity_id") == entity2.get("id") 
                                             for r in cross_relationships)
                            if not exact_exists:
                                relationship = {
                                    "id": f"type_{entity1.get('id')}_{entity2.get('id')}",
                                    "source_entity_id": entity1.get("id"),
                                    "target_entity_id": entity2.get("id"),
                                    "relationship_type": "CROSS_DOCUMENT_TYPE",
                                    "weight": 0.4,
                                    "document_ids": doc_ids,
                                    "metadata": {
                                        "created_by": "link_documents",
                                        "match_type": "entity_type",
                                        "entity_type": entity_type,
                                        "confidence": 0.4
                                    }
                                }
                                cross_relationships.append(relationship)
        
        # Store cross-document relationships
        if cross_relationships:
            await storage_service.store_relationships(cross_relationships)
            await graph_service.store_relationships(cross_relationships)
        
        # Count relationships by type
        exact_count = len([r for r in cross_relationships if r.get("relationship_type") == "CROSS_DOCUMENT_EXACT"])
        fuzzy_count = len([r for r in cross_relationships if r.get("relationship_type") == "CROSS_DOCUMENT_SIMILAR"])
        type_count = len([r for r in cross_relationships if r.get("relationship_type") == "CROSS_DOCUMENT_TYPE"])
        
        return {
            "status": "success", 
            "message": f"Enhanced document linking completed! Found {len(cross_relationships)} cross-document relationships",
            "cross_relationships": len(cross_relationships),
            "entity_matches": entity_matches,
            "breakdown": {
                "exact_matches": exact_count,
                "fuzzy_matches": fuzzy_count, 
                "type_matches": type_count
            },
            "total_entities_processed": len(all_entities),
            "strategies_used": ["exact_name", "fuzzy_name", "entity_type"]
        }
        
    except Exception as e:
        logger.error(f"Error linking documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to link documents: {str(e)}")