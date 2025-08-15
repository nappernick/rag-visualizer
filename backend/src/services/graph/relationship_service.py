"""
Relationship Service - Handles relationship management in the knowledge graph
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import uuid

from ...models.schemas import Relationship

logger = logging.getLogger(__name__)


class RelationshipService:
    """Service for managing relationships between entities"""
    
    def __init__(self, graph_store=None):
        self.graph_store = graph_store
        self.relationship_cache = {}
        
    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
        confidence: float = 0.9
    ) -> Relationship:
        """
        Create a relationship between two entities
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relationship_type: Type of relationship
            properties: Additional properties
            confidence: Confidence score
            
        Returns:
            Created relationship
        """
        relationship = Relationship(
            id=str(uuid.uuid4()),
            source=source_id,
            target=target_id,
            type=relationship_type,
            properties=properties or {},
            confidence=confidence,
            created_at=datetime.now()
        )
        
        # Store in graph database
        if self.graph_store:
            await self.graph_store.create_edge(
                edge_id=relationship.id,
                source_id=source_id,
                target_id=target_id,
                edge_type=relationship_type,
                properties={
                    'confidence': confidence,
                    **(properties or {})
                }
            )
        
        # Cache relationship
        self.relationship_cache[relationship.id] = relationship
        
        return relationship
    
    async def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Get relationship by ID"""
        # Check cache first
        if relationship_id in self.relationship_cache:
            return self.relationship_cache[relationship_id]
        
        # Query graph store
        if self.graph_store:
            edge = await self.graph_store.get_edge(relationship_id)
            if edge:
                relationship = Relationship(
                    id=relationship_id,
                    source=edge.get('source'),
                    target=edge.get('target'),
                    type=edge.get('type'),
                    properties=edge.get('properties', {}),
                    confidence=edge.get('confidence', 0.9)
                )
                self.relationship_cache[relationship_id] = relationship
                return relationship
        
        return None
    
    async def find_relationships_by_entity(
        self,
        entity_id: str,
        direction: str = "both",
        relationship_type: Optional[str] = None
    ) -> List[Relationship]:
        """
        Find relationships connected to an entity
        
        Args:
            entity_id: Entity ID
            direction: "incoming", "outgoing", or "both"
            relationship_type: Optional type filter
            
        Returns:
            List of relationships
        """
        relationships = []
        
        if self.graph_store:
            # Build query based on direction
            if direction == "outgoing":
                query = f"MATCH (n)-[r]->(m) WHERE n.id = '{entity_id}'"
            elif direction == "incoming":
                query = f"MATCH (n)<-[r]-(m) WHERE n.id = '{entity_id}'"
            else:  # both
                query = f"MATCH (n)-[r]-(m) WHERE n.id = '{entity_id}'"
            
            if relationship_type:
                query += f" AND type(r) = '{relationship_type}'"
            
            query += " RETURN r, n.id as source, m.id as target"
            
            results = await self.graph_store.execute_query(query)
            
            for result in results:
                relationship = Relationship(
                    id=result['r']['id'],
                    source=result['source'],
                    target=result['target'],
                    type=result['r']['type'],
                    properties=result['r'].get('properties', {}),
                    confidence=result['r'].get('confidence', 0.9)
                )
                relationships.append(relationship)
        
        return relationships
    
    async def find_relationships_between(
        self,
        source_id: str,
        target_id: str,
        relationship_type: Optional[str] = None
    ) -> List[Relationship]:
        """
        Find relationships between two specific entities
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relationship_type: Optional type filter
            
        Returns:
            List of relationships between the entities
        """
        relationships = []
        
        if self.graph_store:
            query = f"""
            MATCH (n)-[r]->(m) 
            WHERE n.id = '{source_id}' AND m.id = '{target_id}'
            """
            
            if relationship_type:
                query += f" AND type(r) = '{relationship_type}'"
            
            query += " RETURN r"
            
            results = await self.graph_store.execute_query(query)
            
            for result in results:
                relationship = Relationship(
                    id=result['r']['id'],
                    source=source_id,
                    target=target_id,
                    type=result['r']['type'],
                    properties=result['r'].get('properties', {}),
                    confidence=result['r'].get('confidence', 0.9)
                )
                relationships.append(relationship)
        
        return relationships
    
    async def find_paths(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 3,
        relationship_types: Optional[List[str]] = None
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Find paths between two entities
        
        Args:
            start_id: Starting entity ID
            end_id: Ending entity ID
            max_depth: Maximum path length
            relationship_types: Optional list of allowed relationship types
            
        Returns:
            List of paths, where each path is a list of (source, relationship, target) tuples
        """
        paths = []
        
        if self.graph_store:
            rel_filter = ""
            if relationship_types:
                rel_filter = f"[:{':'.join(relationship_types)}]"
            else:
                rel_filter = ""
            
            query = f"""
            MATCH path = (start)-{rel_filter}*1..{max_depth}-(end)
            WHERE start.id = '{start_id}' AND end.id = '{end_id}'
            RETURN path
            LIMIT 10
            """
            
            results = await self.graph_store.execute_query(query)
            
            for result in results:
                path = []
                nodes = result['path']['nodes']
                relationships = result['path']['relationships']
                
                for i in range(len(relationships)):
                    path.append((
                        nodes[i]['id'],
                        relationships[i]['type'],
                        nodes[i + 1]['id']
                    ))
                
                paths.append(path)
        
        return paths
    
    async def update_relationship(
        self,
        relationship_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Relationship]:
        """
        Update relationship properties
        
        Args:
            relationship_id: Relationship ID
            updates: Properties to update
            
        Returns:
            Updated relationship or None if not found
        """
        relationship = await self.get_relationship(relationship_id)
        if not relationship:
            return None
        
        # Update properties
        for key, value in updates.items():
            if hasattr(relationship, key):
                setattr(relationship, key, value)
            else:
                relationship.properties[key] = value
        
        # Update in graph store
        if self.graph_store:
            await self.graph_store.update_edge(relationship_id, updates)
        
        # Update cache
        self.relationship_cache[relationship_id] = relationship
        
        return relationship
    
    async def delete_relationship(self, relationship_id: str) -> bool:
        """
        Delete a relationship
        
        Args:
            relationship_id: Relationship ID
            
        Returns:
            True if deleted, False if not found
        """
        # Remove from cache
        if relationship_id in self.relationship_cache:
            del self.relationship_cache[relationship_id]
        
        # Delete from graph store
        if self.graph_store:
            return await self.graph_store.delete_edge(relationship_id)
        
        return False
    
    async def get_relationship_statistics(
        self,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about relationships
        
        Args:
            document_id: Optional document filter
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_relationships': 0,
            'relationship_types': {},
            'avg_confidence': 0.0,
            'most_connected_entities': []
        }
        
        if self.graph_store:
            query = "MATCH ()-[r]-() "
            if document_id:
                query += f"WHERE r.document_id = '{document_id}' "
            query += "RETURN count(r) as count, type(r) as type, avg(r.confidence) as avg_conf"
            
            results = await self.graph_store.execute_query(query)
            
            for result in results:
                stats['total_relationships'] += result['count']
                stats['relationship_types'][result['type']] = result['count']
                stats['avg_confidence'] = result['avg_conf']
        
        return stats
    
    def clear_cache(self):
        """Clear the relationship cache"""
        self.relationship_cache.clear()