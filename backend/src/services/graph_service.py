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
        """Create optimized indexes for sub-second GraphRAG performance"""
        if not self.initialized:
            return
        
        try:
            with self.driver.session() as session:
                # Primary entity indexes for fast lookups
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
                
                # Performance optimization: Text search index for fuzzy matching
                session.run("""
                    CREATE TEXT INDEX entity_name_text IF NOT EXISTS
                    FOR (e:Entity) ON (e.name)
                """)
                
                # Frequency-based index for high-frequency entity discovery
                session.run("""
                    CREATE INDEX entity_frequency IF NOT EXISTS
                    FOR (e:Entity) ON (e.frequency)
                """)
                
                # Composite index for optimized GraphRAG queries
                session.run("""
                    CREATE INDEX entity_name_type_freq IF NOT EXISTS
                    FOR (e:Entity) ON (e.name, e.type, e.frequency)
                """)
                
                # Document indexes
                session.run("""
                    CREATE INDEX document_id IF NOT EXISTS
                    FOR (d:Document) ON (d.id)
                """)
                
                # Chunk indexes
                session.run("""
                    CREATE INDEX chunk_id IF NOT EXISTS
                    FOR (c:Chunk) ON (c.id)
                """)
                
                # Relationship indexes for graph traversal optimization
                session.run("""
                    CREATE INDEX relationship_weight IF NOT EXISTS
                    FOR ()-[r:RELATED]-() ON (r.weight)
                """)
                session.run("""
                    CREATE INDEX relationship_type IF NOT EXISTS
                    FOR ()-[r:RELATED]-() ON (r.type)
                """)
                
                logger.info("Optimized Neo4j indexes created for sub-second GraphRAG performance")
        except Exception as e:
            logger.error(f"Error creating optimized indexes: {e}")
    
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
                        WHERE toLower(e.name) CONTAINS toLower($search_term)
                          AND e.type = $entity_type
                        RETURN e.id as id, e.name as name, e.type as type,
                               e.frequency as frequency
                        LIMIT $limit
                    """, search_term=query, entity_type=entity_type, limit=limit)
                else:
                    result = session.run("""
                        MATCH (e:Entity)
                        WHERE toLower(e.name) CONTAINS toLower($search_term)
                        RETURN e.id as id, e.name as name, e.type as type,
                               e.frequency as frequency
                        LIMIT $limit
                    """, search_term=query, limit=limit)
                
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
    
    async def production_graph_retrieval(
        self, 
        query: str, 
        limit: int = 10,
        max_hops: int = 2
    ) -> List[Dict]:
        """Production-grade GraphRAG retrieval with multi-strategy scoring"""
        if not self.initialized:
            return []
        
        try:
            with self.driver.session() as session:
                # Fixed optimized production GraphRAG with proper Cypher syntax
                result = session.run("""
                    MATCH (entity:Entity)
                    WHERE toLower(entity.name) = toLower($search_term)
                       OR toLower(entity.name) CONTAINS toLower($search_term)
                       OR toLower($search_term) CONTAINS toLower(entity.name)
                    
                    RETURN entity.id as id,
                           entity.name as name,
                           entity.type as entity_type,
                           COALESCE(entity.frequency, 1) as frequency,
                           CASE 
                              WHEN toLower(entity.name) = toLower($search_term) THEN 0.95 + (COALESCE(entity.frequency, 1) / 100.0)
                              WHEN toLower(entity.name) CONTAINS toLower($search_term) THEN 0.86 + (COALESCE(entity.frequency, 1) / 100.0)
                              WHEN toLower($search_term) CONTAINS toLower(entity.name) THEN 0.80 + (COALESCE(entity.frequency, 1) / 100.0)
                              ELSE 0.75 + (COALESCE(entity.frequency, 1) / 100.0)
                           END as relevance_score,
                           CASE 
                              WHEN toLower(entity.name) = toLower($search_term) THEN 1.0
                              WHEN toLower(entity.name) CONTAINS toLower($search_term) THEN 0.95
                              ELSE 0.9
                           END as confidence,
                           CASE 
                              WHEN toLower(entity.name) = toLower($search_term) THEN 'exact'
                              WHEN toLower(entity.name) CONTAINS toLower($search_term) THEN 'contains'
                              WHEN toLower($search_term) CONTAINS toLower(entity.name) THEN 'contained_by'
                              ELSE 'partial'
                           END as primary_match_type,
                           [CASE 
                              WHEN toLower(entity.name) = toLower($search_term) THEN 'exact'
                              WHEN toLower(entity.name) CONTAINS toLower($search_term) THEN 'contains'
                              WHEN toLower($search_term) CONTAINS toLower(entity.name) THEN 'contained_by'
                              ELSE 'partial'
                           END] as all_match_types,
                           0 as path_length,
                           {
                               match_reason: 'optimized_indexed_search',
                               quality: CASE 
                                  WHEN toLower(entity.name) = toLower($search_term) THEN 'exact'
                                  WHEN toLower(entity.name) CONTAINS toLower($search_term) THEN 'contains'
                                  ELSE 'contained_by'
                               END,
                               strategy_weight: 1.0,
                               index_used: 'entity_name_type_freq',
                               performance_optimized: true
                           } as context
                    ORDER BY relevance_score DESC, COALESCE(entity.frequency, 1) DESC
                    LIMIT $limit
                """, search_term=query, limit=limit)
                
                entities = []
                for record in result:
                    entities.append({
                        "id": record["id"],
                        "name": record["name"],
                        "entity_type": record["entity_type"],
                        "frequency": record["frequency"] or 1,
                        "relevance_score": record["relevance_score"],
                        "confidence": record["confidence"],
                        "match_type": record["primary_match_type"],
                        "all_match_types": record["all_match_types"],
                        "path_length": record["path_length"],
                        "context": record["context"]
                    })
                
                return entities
                
        except Exception as e:
            logger.error(f"Error in production graph retrieval: {e}")
            return []
    
    async def optimized_graph_traversal(
        self, 
        seed_entities: List[str], 
        max_hops: int = 2,
        min_relationship_weight: float = 0.3,
        limit: int = 20
    ) -> List[Dict]:
        """Optimized graph traversal with relationship-aware scoring using indexes"""
        if not self.initialized or not seed_entities:
            return []
        
        try:
            with self.driver.session() as session:
                # Fixed traversal query - use literal hop values instead of parameters
                if max_hops == 1:
                    hop_pattern = "*1"
                elif max_hops == 2:
                    hop_pattern = "*1..2"
                else:
                    hop_pattern = "*1..3"  # Max 3 hops for performance
                
                result = session.run(f"""
                    MATCH (seed:Entity)
                    WHERE seed.id IN $seed_ids
                    
                    OPTIONAL MATCH path = (seed)-[r:RELATED{hop_pattern}]-(related:Entity)
                    WHERE ALL(rel IN relationships(path) WHERE COALESCE(rel.weight, 0.5) >= $min_weight)
                      AND related <> seed
                    
                    WITH related, seed, path,
                         length(path) as path_length,
                         CASE length(path)
                            WHEN 1 THEN 0.8
                            WHEN 2 THEN 0.6
                            ELSE 0.4
                         END as base_traversal_score,
                         reduce(weight_sum = 0.0, rel IN relationships(path) | 
                            weight_sum + COALESCE(rel.weight, 0.5)) / length(path) as avg_path_weight
                    
                    WHERE related IS NOT NULL
                    
                    WITH related, seed, path_length, 
                         base_traversal_score * avg_path_weight as traversal_score,
                         avg_path_weight,
                         COALESCE(related.frequency, 1) as freq
                    
                    WITH related, seed, path_length, traversal_score, avg_path_weight, freq,
                         traversal_score + (freq / 50.0) as final_score,
                         CASE 
                            WHEN avg_path_weight >= 0.7 THEN 0.9
                            WHEN avg_path_weight >= 0.5 THEN 0.8
                            ELSE 0.7
                         END as confidence
                    
                    RETURN DISTINCT 
                           related.id as id,
                           related.name as name,
                           related.type as entity_type,
                           freq as frequency,
                           final_score as relevance_score,
                           confidence,
                           'traversal' as primary_match_type,
                           ['traversal'] as all_match_types,
                           path_length,
                           {{
                               match_reason: 'optimized_graph_traversal',
                               quality: 'traversal',
                               strategy_weight: 0.75,
                               seed_entity: seed.name,
                               avg_path_weight: avg_path_weight,
                               performance_optimized: true,
                               index_used: 'relationship_weight'
                           }} as context
                    ORDER BY final_score DESC, confidence DESC
                    LIMIT $limit
                """, 
                seed_ids=seed_entities, 
                min_weight=min_relationship_weight, 
                limit=limit)
                
                entities = []
                for record in result:
                    entities.append({
                        "id": record["id"],
                        "name": record["name"],
                        "entity_type": record["entity_type"],
                        "frequency": record["frequency"],
                        "relevance_score": record["relevance_score"],
                        "confidence": record["confidence"],
                        "match_type": record["primary_match_type"],
                        "all_match_types": record["all_match_types"],
                        "path_length": record["path_length"],
                        "context": record["context"]
                    })
                
                return entities
                
        except Exception as e:
            logger.error(f"Error in optimized graph traversal: {e}")
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
        """Clear all data from Neo4j with batch deletion for large datasets"""
        if not self.initialized:
            return {"status": "error", "message": "Neo4j not initialized"}
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Try to reconnect first if needed
                try:
                    with self.driver.session() as session:
                        session.run("RETURN 1")
                except:
                    logger.info("Connection lost, attempting to reconnect...")
                    self._reconnect()
                
                # Now proceed with deletion
                with self.driver.session() as session:
                    # Count before deletion
                    result = session.run("MATCH (n) RETURN count(n) as count")
                    node_count = result.single()["count"]
                    
                    result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                    rel_count = result.single()["count"]
                    
                    logger.info(f"Found {node_count} nodes and {rel_count} relationships to delete")
                    
                    if node_count == 0 and rel_count == 0:
                        return {
                            "status": "success",
                            "nodes_deleted": 0,
                            "relationships_deleted": 0,
                            "message": "Neo4j already clean"
                        }
                    
                    # First attempt: Simple DETACH DELETE
                    logger.info("Attempting simple deletion...")
                    session.run("MATCH (n) DETACH DELETE n")
                    
                    # Verify deletion
                    result = session.run("MATCH (n) RETURN count(n) as count")
                    remaining = result.single()["count"]
                    
                    if remaining == 0:
                        logger.info(f"Successfully deleted {node_count} nodes and {rel_count} relationships")
                        return {
                            "status": "success",
                            "nodes_deleted": node_count,
                            "relationships_deleted": rel_count
                        }
                    
                    # If simple deletion failed, try batch deletion
                    logger.warning(f"{remaining} nodes remain, attempting batch deletion...")
                    
                    batch_size = 1000
                    total_deleted = node_count - remaining
                    
                    while True:
                        result = session.run(f"""
                            MATCH (n)
                            WITH n LIMIT {batch_size}
                            DETACH DELETE n
                            RETURN count(n) as deleted
                        """)
                        deleted = result.single()["deleted"]
                        if deleted == 0:
                            break
                        total_deleted += deleted
                        logger.info(f"  Deleted batch of {deleted} nodes")
                    
                    # Final verification
                    result = session.run("MATCH (n) RETURN count(n) as count")
                    final_count = result.single()["count"]
                    
                    if final_count == 0:
                        logger.info(f"Successfully cleared Neo4j after batch deletion")
                        return {
                            "status": "success",
                            "nodes_deleted": node_count,
                            "relationships_deleted": rel_count
                        }
                    else:
                        logger.error(f"Failed to fully clear Neo4j - {final_count} nodes remain")
                        return {
                            "status": "partial",
                            "nodes_deleted": total_deleted,
                            "relationships_deleted": rel_count,
                            "nodes_remaining": final_count
                        }
                    
            except Exception as e:
                logger.warning(f"Neo4j clear attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Error clearing Neo4j after {max_retries} attempts: {e}")
                    return {"status": "error", "message": str(e)}
                # Wait before retry
                import asyncio
                await asyncio.sleep(2)
    
    def _reconnect(self):
        """Reconnect to Neo4j if connection is lost"""
        try:
            if self.driver:
                try:
                    self.driver.close()
                except:
                    pass
            
            # Re-establish connection
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=3600,  # 1 hour
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            
            logger.info("Successfully reconnected to Neo4j")
            
        except Exception as e:
            logger.error(f"Failed to reconnect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()


# Use centralized service manager
from ..core.service_manager import get_graph_service