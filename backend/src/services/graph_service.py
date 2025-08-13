"""
Graph storage service using Neo4j
"""
import os
from typing import List, Dict, Optional, Any, Tuple
import logging
from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable
import json
from .neo4j_connection_manager import get_neo4j_connection_manager

logger = logging.getLogger(__name__)


class GraphService:
    """Service for managing graph storage in Neo4j"""
    
    def __init__(self):
        # Use connection manager for robust connection handling
        self.conn_manager = get_neo4j_connection_manager()
        self.driver = self.conn_manager.get_driver()
        self.initialized = self.conn_manager.enabled
        
        if self.initialized:
            self._create_indexes()
            logger.info("Neo4j graph service initialized successfully")
        else:
            logger.info("Neo4j not configured - graph operations will be disabled")
    
    def _create_indexes(self):
        """Create indexes for better performance"""
        if not self.initialized:
            return
        
        try:
            with self.driver.session() as session:
                # Create indexes for entities
                session.run("""
                    CREATE INDEX entity_id IF NOT EXISTS
                    FOR (e:Entity) ON (e.id)
                """)
                session.run("""
                    CREATE INDEX entity_name IF NOT EXISTS
                    FOR (e:Entity) ON (e.name)
                """)
                session.run("""
                    CREATE INDEX entity_type IF NOT EXISTS
                    FOR (e:Entity) ON (e.type)
                """)
                
                # Create indexes for documents
                session.run("""
                    CREATE INDEX document_id IF NOT EXISTS
                    FOR (d:Document) ON (d.id)
                """)
                
                # Create indexes for chunks
                session.run("""
                    CREATE INDEX chunk_id IF NOT EXISTS
                    FOR (c:Chunk) ON (c.id)
                """)
                
                logger.info("Neo4j indexes created")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    async def store_entities(self, entities: List[Dict]) -> bool:
        """Store entities in Neo4j"""
        if not entities:
            return False
            
        if not self.initialized:
            logger.info("Neo4j not available - skipping entity storage")
            return True  # Return success to not block the flow
        
        try:
            with self.driver.session() as session:
                for entity in entities:
                    session.run("""
                        MERGE (e:Entity {id: $id})
                        SET e.name = $name,
                            e.type = $type,
                            e.frequency = $frequency,
                            e.document_ids = $document_ids,
                            e.chunk_ids = $chunk_ids,
                            e.metadata = $metadata
                    """, 
                    id=entity["id"],
                    name=entity["name"],
                    type=entity.get("entity_type", entity.get("type", "Unknown")),
                    frequency=entity.get("frequency", 1),
                    document_ids=entity.get("document_ids", []),
                    chunk_ids=entity.get("chunk_ids", []),
                    metadata=json.dumps(entity.get("metadata", {}))
                    )
                
                logger.info(f"Stored {len(entities)} entities in Neo4j")
                return True
                
        except Exception as e:
            logger.error(f"Error storing entities: {e}")
            return False
    
    async def store_relationships(self, relationships: List[Dict]) -> bool:
        """Store relationships between entities in Neo4j"""
        if not self.initialized or not relationships:
            return False
        
        try:
            with self.driver.session() as session:
                for rel in relationships:
                    session.run("""
                        MATCH (source:Entity {id: $source_id})
                        MATCH (target:Entity {id: $target_id})
                        MERGE (source)-[r:RELATED {type: $rel_type}]->(target)
                        SET r.weight = $weight,
                            r.document_ids = $document_ids,
                            r.metadata = $metadata
                    """,
                    source_id=rel.get("source_entity_id", rel.get("source_id")),
                    target_id=rel.get("target_entity_id", rel.get("target_id")),
                    rel_type=rel.get("relationship_type", rel.get("type", "related_to")),
                    weight=rel.get("weight", 1.0),
                    document_ids=rel.get("document_ids", []),
                    metadata=json.dumps(rel.get("metadata", {}))
                    )
                
                logger.info(f"Stored {len(relationships)} relationships in Neo4j")
                return True
                
        except Exception as e:
            logger.error(f"Error storing relationships: {e}")
            return False
    
    async def link_entities_to_document(self, document_id: str, entity_ids: List[str]) -> bool:
        """Create links between document and entities"""
        if not self.initialized or not entity_ids:
            return False
        
        try:
            with self.driver.session() as session:
                # First create/update document node
                session.run("""
                    MERGE (d:Document {id: $doc_id})
                """, doc_id=document_id)
                
                # Link entities to document
                for entity_id in entity_ids:
                    session.run("""
                        MATCH (d:Document {id: $doc_id})
                        MATCH (e:Entity {id: $entity_id})
                        MERGE (d)-[:CONTAINS]->(e)
                    """, doc_id=document_id, entity_id=entity_id)
                
                logger.info(f"Linked {len(entity_ids)} entities to document {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error linking entities to document: {e}")
            return False
    
    async def link_entities_to_chunks(self, chunk_id: str, entity_ids: List[str]) -> bool:
        """Create links between chunk and entities"""
        if not self.initialized or not entity_ids:
            return False
        
        try:
            with self.driver.session() as session:
                # First create/update chunk node
                session.run("""
                    MERGE (c:Chunk {id: $chunk_id})
                """, chunk_id=chunk_id)
                
                # Link entities to chunk
                for entity_id in entity_ids:
                    session.run("""
                        MATCH (c:Chunk {id: $chunk_id})
                        MATCH (e:Entity {id: $entity_id})
                        MERGE (c)-[:MENTIONS]->(e)
                    """, chunk_id=chunk_id, entity_id=entity_id)
                
                return True
                
        except Exception as e:
            logger.error(f"Error linking entities to chunk: {e}")
            return False
    
    async def get_document_entities(self, document_id: str) -> List[Dict]:
        """Get all entities for a document"""
        if not self.initialized:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (d:Document {id: $doc_id})-[:CONTAINS]->(e:Entity)
                    RETURN e.id as id, e.name as name, e.type as type,
                           e.frequency as frequency, e.document_ids as document_ids,
                           e.chunk_ids as chunk_ids, e.metadata as metadata
                """, doc_id=document_id)
                
                entities = []
                for record in result:
                    entity = {
                        "id": record["id"],
                        "name": record["name"],
                        "entity_type": record["type"],
                        "frequency": record["frequency"] or 1,
                        "document_ids": record["document_ids"] or [document_id],
                        "chunk_ids": record["chunk_ids"] or [],
                        "metadata": json.loads(record["metadata"]) if record["metadata"] else {}
                    }
                    entities.append(entity)
                
                return entities
                
        except Exception as e:
            logger.error(f"Error getting document entities: {e}")
            return []
    
    async def get_document_relationships(self, document_id: str) -> List[Dict]:
        """Get all relationships for entities in a document"""
        if not self.initialized:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (d:Document {id: $doc_id})-[:CONTAINS]->(e1:Entity)
                    MATCH (d)-[:CONTAINS]->(e2:Entity)
                    MATCH (e1)-[r:RELATED]->(e2)
                    RETURN DISTINCT 
                           e1.id as source_id, 
                           e2.id as target_id,
                           r.type as type,
                           r.weight as weight,
                           r.document_ids as document_ids,
                           r.metadata as metadata
                """, doc_id=document_id)
                
                relationships = []
                for record in result:
                    rel = {
                        "id": f"{record['source_id']}_{record['target_id']}",
                        "source_entity_id": record["source_id"],
                        "target_entity_id": record["target_id"],
                        "relationship_type": record["type"] or "related_to",
                        "weight": record["weight"] or 1.0,
                        "document_ids": record["document_ids"] or [document_id],
                        "metadata": json.loads(record["metadata"]) if record["metadata"] else {}
                    }
                    relationships.append(rel)
                
                return relationships
                
        except Exception as e:
            logger.error(f"Error getting document relationships: {e}")
            return []
    
    async def search_entities(
        self, 
        query: str, 
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Search for entities by name or type"""
        if not self.initialized:
            return []
        
        try:
            with self.driver.session() as session:
                if entity_type:
                    result = session.run("""
                        MATCH (e:Entity)
                        WHERE toLower(e.name) CONTAINS toLower($query)
                          AND e.type = $entity_type
                        RETURN e.id as id, e.name as name, e.type as type,
                               e.frequency as frequency
                        LIMIT $limit
                    """, query=query, entity_type=entity_type, limit=limit)
                else:
                    result = session.run("""
                        MATCH (e:Entity)
                        WHERE toLower(e.name) CONTAINS toLower($query)
                        RETURN e.id as id, e.name as name, e.type as type,
                               e.frequency as frequency
                        LIMIT $limit
                    """, query=query, limit=limit)
                
                entities = []
                for record in result:
                    entities.append({
                        "id": record["id"],
                        "name": record["name"],
                        "entity_type": record["type"],
                        "frequency": record["frequency"] or 1
                    })
                
                return entities
                
        except Exception as e:
            logger.error(f"Error searching entities: {e}")
            return []
    
    async def get_related_entities(
        self, 
        entity_id: str, 
        max_hops: int = 2
    ) -> List[Tuple[Dict, int]]:
        """Get entities related to a given entity within max_hops"""
        if not self.initialized:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH path = (start:Entity {id: $entity_id})-[:RELATED*1..$max_hops]-(related:Entity)
                    WITH related, min(length(path)) as distance
                    RETURN DISTINCT 
                           related.id as id, 
                           related.name as name, 
                           related.type as type,
                           distance
                    ORDER BY distance
                    LIMIT 50
                """, entity_id=entity_id, max_hops=max_hops)
                
                entities = []
                for record in result:
                    entity = {
                        "id": record["id"],
                        "name": record["name"],
                        "entity_type": record["type"]
                    }
                    entities.append((entity, record["distance"]))
                
                return entities
                
        except Exception as e:
            logger.error(f"Error getting related entities: {e}")
            return []
    
    async def delete_document_graph(self, document_id: str) -> bool:
        """Delete document and its relationships from graph"""
        if not self.initialized:
            return False
        
        try:
            with self.driver.session() as session:
                # Delete document node and relationships
                session.run("""
                    MATCH (d:Document {id: $doc_id})
                    DETACH DELETE d
                """, doc_id=document_id)
                
                # Clean up orphaned entities (entities with no document connections)
                session.run("""
                    MATCH (e:Entity)
                    WHERE NOT ((:Document)-[:CONTAINS]->(e))
                    DETACH DELETE e
                """)
                
                logger.info(f"Deleted graph data for document {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting document graph: {e}")
            return False
    
    async def clear_all(self) -> Dict:
        """Clear all data from Neo4j"""
        if not self.initialized:
            return {"status": "error", "message": "Neo4j not initialized"}
        
        try:
            with self.driver.session() as session:
                # Count before deletion
                result = session.run("""
                    MATCH (n)
                    RETURN count(n) as node_count
                """)
                node_count = result.single()["node_count"]
                
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN count(r) as rel_count
                """)
                rel_count = result.single()["rel_count"]
                
                # Delete everything
                session.run("""
                    MATCH (n)
                    DETACH DELETE n
                """)
                
                logger.info(f"Cleared {node_count} nodes and {rel_count} relationships from Neo4j")
                
                return {
                    "status": "success",
                    "nodes_deleted": node_count,
                    "relationships_deleted": rel_count
                }
                
        except Exception as e:
            logger.error(f"Error clearing Neo4j: {e}")
            return {"status": "error", "message": str(e)}
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()


# Use centralized service manager
from ..core.service_manager import get_graph_service