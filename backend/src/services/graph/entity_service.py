"""
Entity Service - Handles entity CRUD operations
"""
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import uuid
from sqlalchemy.orm import Session

from ...models.schemas import Entity, EntityCreate
from ...database import get_session

logger = logging.getLogger(__name__)


class EntityService:
    """Service for managing entities in the knowledge graph"""
    
    def __init__(self, graph_store=None):
        self.graph_store = graph_store
        self.entity_cache = {}
        
    async def create_entity(
        self,
        name: str,
        entity_type: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 0.9
    ) -> Entity:
        """
        Create a new entity
        
        Args:
            name: Entity name
            entity_type: Type of entity (person, organization, location, etc.)
            document_id: Associated document ID
            metadata: Additional metadata
            confidence: Confidence score
            
        Returns:
            Created entity
        """
        entity = Entity(
            id=str(uuid.uuid4()),
            name=name,
            type=entity_type,
            document_id=document_id,
            metadata=metadata or {},
            confidence=confidence,
            created_at=datetime.now()
        )
        
        # Store in graph database if available
        if self.graph_store:
            await self.graph_store.create_node(
                node_id=entity.id,
                labels=[entity_type],
                properties={
                    'name': name,
                    'document_id': document_id,
                    'confidence': confidence,
                    **metadata
                }
            )
        
        # Cache entity
        self.entity_cache[entity.id] = entity
        
        return entity
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        # Check cache first
        if entity_id in self.entity_cache:
            return self.entity_cache[entity_id]
        
        # Query graph store
        if self.graph_store:
            node = await self.graph_store.get_node(entity_id)
            if node:
                entity = Entity(
                    id=entity_id,
                    name=node.get('name'),
                    type=node.get('labels', ['unknown'])[0],
                    document_id=node.get('document_id'),
                    metadata=node.get('properties', {}),
                    confidence=node.get('confidence', 0.9)
                )
                self.entity_cache[entity_id] = entity
                return entity
        
        return None
    
    async def find_entities_by_name(
        self,
        name: str,
        entity_type: Optional[str] = None,
        fuzzy: bool = False
    ) -> List[Entity]:
        """
        Find entities by name
        
        Args:
            name: Entity name to search for
            entity_type: Optional type filter
            fuzzy: Whether to use fuzzy matching
            
        Returns:
            List of matching entities
        """
        entities = []
        
        if self.graph_store:
            query = f"MATCH (n) WHERE n.name "
            if fuzzy:
                query += f"CONTAINS '{name}'"
            else:
                query += f"= '{name}'"
            
            if entity_type:
                query += f" AND '{entity_type}' IN labels(n)"
            
            query += " RETURN n"
            
            results = await self.graph_store.execute_query(query)
            
            for result in results:
                entity = Entity(
                    id=result['n']['id'],
                    name=result['n']['name'],
                    type=result['n'].get('labels', ['unknown'])[0],
                    document_id=result['n'].get('document_id'),
                    metadata=result['n'].get('properties', {}),
                    confidence=result['n'].get('confidence', 0.9)
                )
                entities.append(entity)
        
        return entities
    
    async def find_entities_by_document(
        self,
        document_id: str,
        entity_type: Optional[str] = None
    ) -> List[Entity]:
        """
        Find all entities in a document
        
        Args:
            document_id: Document ID
            entity_type: Optional type filter
            
        Returns:
            List of entities in the document
        """
        entities = []
        
        if self.graph_store:
            query = f"MATCH (n) WHERE n.document_id = '{document_id}'"
            
            if entity_type:
                query += f" AND '{entity_type}' IN labels(n)"
            
            query += " RETURN n"
            
            results = await self.graph_store.execute_query(query)
            
            for result in results:
                entity = Entity(
                    id=result['n']['id'],
                    name=result['n']['name'],
                    type=result['n'].get('labels', ['unknown'])[0],
                    document_id=document_id,
                    metadata=result['n'].get('properties', {}),
                    confidence=result['n'].get('confidence', 0.9)
                )
                entities.append(entity)
        
        return entities
    
    async def update_entity(
        self,
        entity_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Entity]:
        """
        Update entity properties
        
        Args:
            entity_id: Entity ID
            updates: Properties to update
            
        Returns:
            Updated entity or None if not found
        """
        entity = await self.get_entity(entity_id)
        if not entity:
            return None
        
        # Update properties
        for key, value in updates.items():
            if hasattr(entity, key):
                setattr(entity, key, value)
            else:
                entity.metadata[key] = value
        
        # Update in graph store
        if self.graph_store:
            await self.graph_store.update_node(entity_id, updates)
        
        # Update cache
        self.entity_cache[entity_id] = entity
        
        return entity
    
    async def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity
        
        Args:
            entity_id: Entity ID
            
        Returns:
            True if deleted, False if not found
        """
        # Remove from cache
        if entity_id in self.entity_cache:
            del self.entity_cache[entity_id]
        
        # Delete from graph store
        if self.graph_store:
            return await self.graph_store.delete_node(entity_id)
        
        return False
    
    async def merge_entities(
        self,
        entity_ids: List[str],
        merged_name: str
    ) -> Optional[Entity]:
        """
        Merge multiple entities into one
        
        Args:
            entity_ids: List of entity IDs to merge
            merged_name: Name for the merged entity
            
        Returns:
            Merged entity or None if merge failed
        """
        if len(entity_ids) < 2:
            return None
        
        # Get all entities
        entities = []
        for entity_id in entity_ids:
            entity = await self.get_entity(entity_id)
            if entity:
                entities.append(entity)
        
        if not entities:
            return None
        
        # Create merged entity
        merged_metadata = {}
        for entity in entities:
            merged_metadata.update(entity.metadata)
        
        merged_entity = await self.create_entity(
            name=merged_name,
            entity_type=entities[0].type,  # Use first entity's type
            document_id=entities[0].document_id,
            metadata=merged_metadata,
            confidence=max(e.confidence for e in entities)
        )
        
        # Update relationships to point to merged entity
        if self.graph_store:
            for entity_id in entity_ids[1:]:  # Keep first, delete others
                # Transfer relationships
                await self.graph_store.transfer_relationships(
                    from_node=entity_id,
                    to_node=merged_entity.id
                )
                # Delete old entity
                await self.delete_entity(entity_id)
        
        return merged_entity
    
    def clear_cache(self):
        """Clear the entity cache"""
        self.entity_cache.clear()