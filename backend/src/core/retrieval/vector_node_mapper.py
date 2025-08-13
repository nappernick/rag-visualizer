"""
Vector-to-Node mapping for bridging vector search with graph traversal
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from ...services.id_mapper import IDMapper
from ...db import get_session

logger = logging.getLogger(__name__)


class VectorNodeMapper:
    """Maps vector search results to graph nodes for hybrid retrieval."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        vector_config = config.get('vector_store', {})
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=vector_config.get('host', 'localhost'),
            port=vector_config.get('port', 6333)
        )
        self.collection_name = vector_config.get('collection_name', 'rag_chunks')
        
        # Graph configuration
        graph_config = config.get('graph_store', {})
        self.neo4j_enabled = graph_config.get('provider') == 'neo4j'
        
    def map_vectors_to_nodes(self, vector_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Map vector IDs to their corresponding graph nodes.
        
        Args:
            vector_ids: List of Qdrant point IDs
            
        Returns:
            List of graph node information
        """
        graph_nodes = []
        
        with get_session() as session:
            mapper = IDMapper(session)
            
            for vector_id in vector_ids:
                # Get chunk ID from vector ID
                related = mapper.traverse("vector", vector_id)
                chunk_ids = related.get("chunk", [])
                
                if not chunk_ids:
                    continue
                    
                chunk_id = chunk_ids[0]  # Vector should map to exactly one chunk
                
                # Get graph nodes for this chunk
                node_ids = mapper.get_graph_nodes_for_chunk(chunk_id)
                
                # Get entities for this chunk
                entity_ids = mapper.get_entities_for_chunk(chunk_id)
                
                graph_nodes.append({
                    'vector_id': vector_id,
                    'chunk_id': chunk_id,
                    'graph_node_ids': node_ids,
                    'entity_ids': entity_ids
                })
        
        return graph_nodes
    
    def get_seed_nodes_from_search(self, query_embedding: List[float], k: int = 5) -> List[str]:
        """
        Perform vector search and extract seed nodes for graph traversal.
        
        Args:
            query_embedding: Query vector
            k: Number of top results to use as seeds
            
        Returns:
            List of entity/node IDs to use as graph traversal seeds
        """
        # Perform vector search
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k
        )
        
        seed_nodes = []
        seed_entities = set()
        
        with get_session() as session:
            mapper = IDMapper(session)
            
            for result in search_results:
                # Get chunk ID from vector
                chunk_id = result.payload.get('chunk_id')
                if not chunk_id:
                    # Try to get from ID mapping
                    related = mapper.traverse("vector", str(result.id))
                    chunk_ids = related.get("chunk", [])
                    if chunk_ids:
                        chunk_id = chunk_ids[0]
                
                if chunk_id:
                    # Get entities for this chunk
                    entities = mapper.get_entities_for_chunk(chunk_id)
                    seed_entities.update(entities)
                    
                    # Get graph nodes if available
                    graph_nodes = mapper.get_graph_nodes_for_chunk(chunk_id)
                    seed_nodes.extend(graph_nodes)
        
        # Combine entities and graph nodes as seeds
        all_seeds = list(seed_entities) + seed_nodes
        
        logger.info(f"Found {len(all_seeds)} seed nodes from {k} vector results")
        return all_seeds
    
    def expand_from_seed_nodes(self, node_ids: List[str], depth: int = 2) -> List[Dict[str, Any]]:
        """
        Expand from seed nodes to find related nodes/entities.
        
        Args:
            node_ids: Initial seed node IDs
            depth: How many hops to expand
            
        Returns:
            List of expanded graph nodes with relevance scores
        """
        expanded_nodes = []
        visited = set(node_ids)
        current_level = node_ids
        
        with get_session() as session:
            mapper = IDMapper(session)
            
            for level in range(depth):
                next_level = []
                level_weight = 1.0 / (level + 1)  # Decay weight by distance
                
                for node_id in current_level:
                    # Check if it's an entity
                    related_entities = mapper.get_related_entities(node_id, max_depth=1)
                    
                    for depth_key, entity_list in related_entities.items():
                        for entity_id in entity_list:
                            if entity_id not in visited:
                                visited.add(entity_id)
                                next_level.append(entity_id)
                                
                                # Get chunks for this entity
                                chunks = mapper.get_chunks_for_entity(entity_id)
                                
                                expanded_nodes.append({
                                    'node_id': entity_id,
                                    'node_type': 'entity',
                                    'distance': level + 1,
                                    'weight': level_weight,
                                    'chunk_ids': chunks
                                })
                
                current_level = next_level
                if not current_level:
                    break
        
        return expanded_nodes
    
    def update_vector_payloads_with_mappings(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Update Qdrant vector payloads with entity and graph node IDs.
        
        Args:
            chunks: List of chunk dictionaries with mapping information
        """
        points_to_update = []
        
        with get_session() as session:
            mapper = IDMapper(session)
            
            for chunk in chunks:
                chunk_id = chunk['id']
                vector_id = chunk.get('vector_id') or mapper.get_vector_for_chunk(chunk_id)
                
                if not vector_id:
                    continue
                
                # Get all related IDs
                entity_ids = mapper.get_entities_for_chunk(chunk_id)
                graph_node_ids = mapper.get_graph_nodes_for_chunk(chunk_id)
                
                # Prepare updated payload
                payload = {
                    'chunk_id': chunk_id,
                    'document_id': chunk.get('document_id'),
                    'entity_ids': entity_ids,
                    'graph_node_ids': graph_node_ids,
                    'content': chunk.get('content', ''),
                    'metadata': chunk.get('metadata', {})
                }
                
                points_to_update.append(
                    PointStruct(
                        id=vector_id,
                        vector=chunk.get('embedding', []),
                        payload=payload
                    )
                )
        
        # Batch update Qdrant
        if points_to_update:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points_to_update
            )
            logger.info(f"Updated {len(points_to_update)} vector payloads with mappings")
    
    def get_chunks_from_vector_search(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        include_graph_context: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search and enrich results with graph context.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            include_graph_context: Whether to include graph node/entity info
            
        Returns:
            Enriched chunk results with graph context
        """
        # Perform vector search
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            with_payload=True
        )
        
        enriched_results = []
        
        for result in search_results:
            chunk_data = {
                'chunk_id': result.payload.get('chunk_id', str(result.id)),
                'content': result.payload.get('content', ''),
                'score': result.score,
                'document_id': result.payload.get('document_id'),
                'metadata': result.payload.get('metadata', {})
            }
            
            if include_graph_context:
                chunk_data['entity_ids'] = result.payload.get('entity_ids', [])
                chunk_data['graph_node_ids'] = result.payload.get('graph_node_ids', [])
            
            enriched_results.append(chunk_data)
        
        return enriched_results
    
    def compute_hybrid_scores(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        vector_weight: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Compute hybrid scores combining vector and graph results.
        
        Args:
            vector_results: Results from vector search
            graph_results: Results from graph traversal
            vector_weight: Weight for vector scores (graph weight = 1 - vector_weight)
            
        Returns:
            List of (chunk_id, hybrid_score) tuples
        """
        chunk_scores = {}
        
        # Process vector results
        for result in vector_results:
            chunk_id = result['chunk_id']
            chunk_scores[chunk_id] = {
                'vector_score': result['score'],
                'graph_score': 0.0
            }
        
        # Process graph results
        for result in graph_results:
            for chunk_id in result.get('chunk_ids', []):
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = {
                        'vector_score': 0.0,
                        'graph_score': 0.0
                    }
                # Graph score based on distance/weight
                chunk_scores[chunk_id]['graph_score'] = max(
                    chunk_scores[chunk_id]['graph_score'],
                    result.get('weight', 0.5)
                )
        
        # Compute hybrid scores
        hybrid_scores = []
        for chunk_id, scores in chunk_scores.items():
            hybrid_score = (
                vector_weight * scores['vector_score'] +
                (1 - vector_weight) * scores['graph_score']
            )
            hybrid_scores.append((chunk_id, hybrid_score))
        
        # Sort by score
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        
        return hybrid_scores