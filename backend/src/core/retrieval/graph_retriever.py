"""
Graph-based retrieval using Neo4j and entity relationships
"""
from typing import List, Dict, Any, Optional
import logging
from neo4j import GraphDatabase

from ...models import RetrievalResult
from ...services.id_mapper import IDMapper
from ...db import get_session

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Handles graph-based retrieval from Neo4j"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        graph_config = config.get('graph_store', {})
        
        # Initialize Neo4j driver if configured
        self.driver = None
        if graph_config.get('provider') == 'neo4j':
            try:
                self.driver = GraphDatabase.driver(
                    graph_config.get('uri', 'bolt://localhost:7687'),
                    auth=(
                        graph_config.get('username', 'neo4j'),
                        graph_config.get('password', 'password')
                    )
                )
                logger.info("Connected to Neo4j")
            except Exception as e:
                logger.warning(f"Failed to connect to Neo4j: {e}. Using fallback retrieval.")
    
    def retrieve(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks based on graph traversal
        
        Args:
            query: Text query
            k: Number of results to return
            
        Returns:
            List of retrieval results
        """
        # Extract entities from query (simplified - in production use NER)
        query_entities = self._extract_query_entities(query)
        
        if not query_entities:
            return []
        
        # Find seed nodes based on query entities
        seed_nodes = self._find_seed_nodes(query_entities)
        
        # Traverse graph from seed nodes
        return self.retrieve_from_seeds(seed_nodes, query, k)
    
    def retrieve_from_seeds(
        self,
        seed_nodes: List[str],
        query: Optional[str] = None,
        max_depth: int = 2,
        confidence_threshold: float = 0.6,
        max_nodes: int = 100
    ) -> List[RetrievalResult]:
        """
        Retrieve chunks by traversing from seed nodes
        
        Args:
            seed_nodes: Initial entity/node IDs to start from
            query: Optional query text for relevance scoring
            max_depth: Maximum traversal depth
            confidence_threshold: Minimum confidence for edges
            max_nodes: Maximum nodes to explore
            
        Returns:
            List of retrieval results
        """
        results = []
        visited = set()
        
        with get_session() as session:
            mapper = IDMapper(session)
            
            # BFS traversal from seed nodes
            current_level = seed_nodes
            for depth in range(max_depth):
                if not current_level or len(visited) >= max_nodes:
                    break
                
                next_level = []
                level_weight = 1.0 / (depth + 1)
                
                for node_id in current_level:
                    if node_id in visited:
                        continue
                    visited.add(node_id)
                    
                    # Get chunks for this entity/node
                    chunks = mapper.get_chunks_for_entity(node_id)
                    
                    for chunk_id in chunks:
                        # Create retrieval result
                        result = RetrievalResult(
                            chunk_id=chunk_id,
                            content=self._get_chunk_content(chunk_id),
                            score=level_weight * confidence_threshold,
                            source="graph",
                            metadata={
                                "entity_id": node_id,
                                "traversal_depth": depth,
                                "confidence": confidence_threshold
                            },
                            highlights=[]
                        )
                        results.append(result)
                    
                    # Get related entities for next level
                    if depth < max_depth - 1:
                        related = mapper.get_related_entities(node_id, max_depth=1)
                        for rel_depth, entities in related.items():
                            next_level.extend(entities)
                
                current_level = next_level
        
        # Sort by score and deduplicate
        seen_chunks = {}
        for result in results:
            if result.chunk_id not in seen_chunks or result.score > seen_chunks[result.chunk_id].score:
                seen_chunks[result.chunk_id] = result
        
        final_results = list(seen_chunks.values())
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return final_results
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """
        Extract entities from query text
        In production, use NER or entity linking
        """
        # Simplified: look for capitalized words and known entity patterns
        entities = []
        words = query.split()
        
        for word in words:
            # Simple heuristic: capitalized words might be entities
            if word[0].isupper() and len(word) > 2:
                entities.append(word.lower())
        
        return entities
    
    def _find_seed_nodes(self, query_entities: List[str]) -> List[str]:
        """
        Find entity/node IDs matching query entities
        """
        seed_nodes = []
        
        # In production, query Neo4j or entity index
        # For now, return simplified entity IDs
        with get_session() as session:
            # This would query the actual entity storage
            # Simplified version:
            for entity_name in query_entities:
                # Would normally search for entities by name
                seed_nodes.append(f"entity_{entity_name}")
        
        return seed_nodes
    
    def _get_chunk_content(self, chunk_id: str) -> str:
        """
        Retrieve chunk content by ID
        In production, fetch from storage
        """
        # Simplified - would fetch from PostgreSQL or document store
        return f"Content for chunk {chunk_id}"
    
    def get_entity_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        Get relationships for a specific entity
        """
        if not self.driver:
            return []
        
        relationships = []
        
        with self.driver.session() as session:
            query = """
            MATCH (e:Entity {id: $entity_id})-[r]-(related:Entity)
            RETURN related.id as related_id, 
                   related.name as related_name,
                   type(r) as relationship_type,
                   r.weight as weight
            LIMIT 50
            """
            
            result = session.run(query, entity_id=entity_id)
            
            for record in result:
                relationships.append({
                    "related_id": record["related_id"],
                    "related_name": record["related_name"],
                    "relationship_type": record["relationship_type"],
                    "weight": record["weight"] or 1.0
                })
        
        return relationships
    
    def find_path_between_entities(
        self,
        start_entity: str,
        end_entity: str,
        max_hops: int = 3
    ) -> List[List[str]]:
        """
        Find paths between two entities in the graph
        """
        if not self.driver:
            return []
        
        paths = []
        
        with self.driver.session() as session:
            query = f"""
            MATCH path = shortestPath(
                (start:Entity {{id: $start_id}})-[*..{max_hops}]-(end:Entity {{id: $end_id}})
            )
            RETURN [n in nodes(path) | n.id] as node_ids
            LIMIT 5
            """
            
            result = session.run(
                query,
                start_id=start_entity,
                end_id=end_entity
            )
            
            for record in result:
                paths.append(record["node_ids"])
        
        return paths
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph
        """
        if not self.driver:
            return {"status": "Neo4j not connected"}
        
        stats = {}
        
        with self.driver.session() as session:
            # Count entities
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            stats["entity_count"] = result.single()["count"]
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            stats["relationship_count"] = result.single()["count"]
            
            # Get relationship types
            result = session.run("MATCH ()-[r]->() RETURN DISTINCT type(r) as type")
            stats["relationship_types"] = [record["type"] for record in result]
        
        return stats
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()