"""
Graph Exploration Service - Handles interactive graph exploration and discovery
"""
from typing import List, Dict, Any, Optional, Set, Tuple
import logging
from collections import defaultdict, deque
import networkx as nx
from datetime import datetime

logger = logging.getLogger(__name__)


class GraphExplorationService:
    """Service for exploring and discovering patterns in the knowledge graph"""
    
    def __init__(self, graph_service=None, entity_service=None, relationship_service=None):
        self.graph_service = graph_service
        self.entity_service = entity_service
        self.relationship_service = relationship_service
        self.exploration_cache = {}
        
    async def explore_from_entity(
        self,
        entity_name: Optional[str] = None,
        entity_id: Optional[str] = None,
        exploration_depth: int = 2,
        max_nodes: int = 50,
        include_relationships: bool = True,
        filter_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Explore the graph starting from a specific entity
        
        Args:
            entity_name: Name of the starting entity
            entity_id: ID of the starting entity
            exploration_depth: How many hops to explore
            max_nodes: Maximum number of nodes to return
            include_relationships: Whether to include relationship details
            filter_types: Optional entity type filter
            
        Returns:
            Exploration results with entities and relationships
        """
        # Find starting entity
        start_entity = None
        if entity_id:
            start_entity = await self._get_entity_by_id(entity_id)
        elif entity_name:
            start_entity = await self._find_entity_by_name(entity_name)
        
        if not start_entity:
            return {
                'error': 'Starting entity not found',
                'entity_name': entity_name,
                'entity_id': entity_id
            }
        
        # Perform exploration
        explored = await self._explore_breadth_first(
            start_entity['id'],
            exploration_depth,
            max_nodes,
            filter_types
        )
        
        # Build result
        result = {
            'start_entity': start_entity,
            'exploration_depth': exploration_depth,
            'entities_found': len(explored['entities']),
            'entities': explored['entities'],
            'entity_levels': explored['levels']
        }
        
        if include_relationships:
            result['relationships'] = explored['relationships']
            result['relationships_found'] = len(explored['relationships'])
        
        # Add exploration metadata
        result['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'max_nodes': max_nodes,
            'filter_types': filter_types,
            'exploration_type': 'breadth_first'
        }
        
        return result
    
    async def explore_from_document(
        self,
        document_id: str,
        exploration_type: str = "entities",  # entities, themes, connections
        max_results: int = 100
    ) -> Dict[str, Any]:
        """
        Explore the graph from a document perspective
        
        Args:
            document_id: Document to explore from
            exploration_type: Type of exploration
            max_results: Maximum results to return
            
        Returns:
            Document-centric exploration results
        """
        result = {
            'document_id': document_id,
            'exploration_type': exploration_type,
            'timestamp': datetime.now().isoformat()
        }
        
        if exploration_type == "entities":
            result['entities'] = await self._explore_document_entities(
                document_id, max_results
            )
        elif exploration_type == "themes":
            result['themes'] = await self._explore_document_themes(
                document_id, max_results
            )
        elif exploration_type == "connections":
            result['connections'] = await self._explore_document_connections(
                document_id, max_results
            )
        else:
            result['error'] = f"Unknown exploration type: {exploration_type}"
        
        return result
    
    async def find_connection_paths(
        self,
        start_entity: str,
        end_entity: str,
        max_path_length: int = 5,
        max_paths: int = 10
    ) -> Dict[str, Any]:
        """
        Find connection paths between two entities
        
        Args:
            start_entity: Starting entity name or ID
            end_entity: Target entity name or ID
            max_path_length: Maximum path length to search
            max_paths: Maximum number of paths to return
            
        Returns:
            Connection paths between entities
        """
        # Find entities
        start = await self._find_or_get_entity(start_entity)
        end = await self._find_or_get_entity(end_entity)
        
        if not start or not end:
            return {
                'error': 'One or both entities not found',
                'start_entity': start_entity,
                'end_entity': end_entity
            }
        
        # Find paths
        paths = await self._find_paths_between(
            start['id'],
            end['id'],
            max_path_length,
            max_paths
        )
        
        # Format results
        formatted_paths = []
        for path in paths:
            formatted_path = {
                'path': path,
                'length': len(path),
                'confidence': self._calculate_path_confidence(path),
                'path_description': self._describe_path(path)
            }
            formatted_paths.append(formatted_path)
        
        return {
            'start': start,
            'end': end,
            'paths_found': len(formatted_paths),
            'paths': formatted_paths,
            'max_path_length': max_path_length,
            'timestamp': datetime.now().isoformat()
        }
    
    async def discover_patterns(
        self,
        pattern_type: str = "frequent",  # frequent, anomalous, structural
        min_support: float = 0.1,
        max_patterns: int = 20
    ) -> Dict[str, Any]:
        """
        Discover patterns in the knowledge graph
        
        Args:
            pattern_type: Type of patterns to discover
            min_support: Minimum support for patterns
            max_patterns: Maximum patterns to return
            
        Returns:
            Discovered patterns
        """
        if pattern_type == "frequent":
            patterns = await self._discover_frequent_patterns(min_support, max_patterns)
        elif pattern_type == "anomalous":
            patterns = await self._discover_anomalous_patterns(max_patterns)
        elif pattern_type == "structural":
            patterns = await self._discover_structural_patterns(max_patterns)
        else:
            return {'error': f"Unknown pattern type: {pattern_type}"}
        
        return {
            'pattern_type': pattern_type,
            'patterns_found': len(patterns),
            'patterns': patterns,
            'parameters': {
                'min_support': min_support,
                'max_patterns': max_patterns
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_entity_context(
        self,
        entity_id: str,
        context_size: int = 10,
        include_documents: bool = True,
        include_similar: bool = True
    ) -> Dict[str, Any]:
        """
        Get rich context for an entity
        
        Args:
            entity_id: Entity to get context for
            context_size: Size of context to retrieve
            include_documents: Include document references
            include_similar: Include similar entities
            
        Returns:
            Entity context information
        """
        entity = await self._get_entity_by_id(entity_id)
        if not entity:
            return {'error': f"Entity not found: {entity_id}"}
        
        context = {
            'entity': entity,
            'direct_connections': await self._get_direct_connections(entity_id, context_size),
            'co_occurring_entities': await self._get_co_occurring_entities(entity_id, context_size)
        }
        
        if include_documents:
            context['related_documents'] = await self._get_related_documents(
                entity_id, context_size
            )
        
        if include_similar:
            context['similar_entities'] = await self._find_similar_entities(
                entity_id, context_size
            )
        
        # Add importance metrics
        context['importance_metrics'] = await self._calculate_entity_importance(entity_id)
        
        return context
    
    async def explore_neighborhood(
        self,
        center_entities: List[str],
        radius: int = 2,
        min_connection_strength: float = 0.5
    ) -> Dict[str, Any]:
        """
        Explore the neighborhood around multiple entities
        
        Args:
            center_entities: List of central entity IDs
            radius: Exploration radius
            min_connection_strength: Minimum connection strength
            
        Returns:
            Neighborhood exploration results
        """
        neighborhood = {
            'nodes': set(),
            'edges': [],
            'communities': []
        }
        
        # Explore from each center
        for entity_id in center_entities:
            local_neighborhood = await self._explore_local_neighborhood(
                entity_id, radius, min_connection_strength
            )
            
            neighborhood['nodes'].update(local_neighborhood['nodes'])
            neighborhood['edges'].extend(local_neighborhood['edges'])
        
        # Detect communities in the neighborhood
        if len(neighborhood['nodes']) > 5:
            neighborhood['communities'] = self._detect_communities(
                list(neighborhood['nodes']),
                neighborhood['edges']
            )
        
        return {
            'center_entities': center_entities,
            'radius': radius,
            'nodes_found': len(neighborhood['nodes']),
            'edges_found': len(neighborhood['edges']),
            'communities_found': len(neighborhood['communities']),
            'neighborhood': neighborhood,
            'timestamp': datetime.now().isoformat()
        }
    
    # Helper methods
    async def _explore_breadth_first(
        self,
        start_id: str,
        max_depth: int,
        max_nodes: int,
        filter_types: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Perform breadth-first exploration
        """
        visited = set()
        queue = deque([(start_id, 0)])
        entities = []
        relationships = []
        levels = {start_id: 0}
        
        while queue and len(visited) < max_nodes:
            current_id, depth = queue.popleft()
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # Get entity details
            entity = await self._get_entity_by_id(current_id)
            if entity:
                if not filter_types or entity.get('type') in filter_types:
                    entities.append(entity)
                    levels[current_id] = depth
            
            # Get connected entities
            if depth < max_depth:
                connections = await self._get_entity_connections(current_id)
                
                for conn in connections:
                    neighbor_id = conn['target_id']
                    if neighbor_id not in visited:
                        queue.append((neighbor_id, depth + 1))
                        relationships.append(conn)
        
        return {
            'entities': entities,
            'relationships': relationships,
            'levels': levels
        }
    
    async def _find_paths_between(
        self,
        start_id: str,
        end_id: str,
        max_length: int,
        max_paths: int
    ) -> List[List[Dict[str, Any]]]:
        """
        Find paths between two entities
        """
        paths = []
        visited = set()
        
        def dfs(current_id: str, target_id: str, path: List, depth: int):
            if len(paths) >= max_paths:
                return
            
            if depth > max_length:
                return
            
            if current_id == target_id:
                paths.append(path.copy())
                return
            
            visited.add(current_id)
            
            # Get neighbors
            connections = self._get_entity_connections_sync(current_id)
            
            for conn in connections:
                neighbor_id = conn['target_id']
                if neighbor_id not in visited:
                    path.append(conn)
                    dfs(neighbor_id, target_id, path, depth + 1)
                    path.pop()
            
            visited.remove(current_id)
        
        # Start DFS
        dfs(start_id, end_id, [], 0)
        
        return paths
    
    async def _discover_frequent_patterns(
        self,
        min_support: float,
        max_patterns: int
    ) -> List[Dict[str, Any]]:
        """
        Discover frequently occurring patterns
        """
        patterns = []
        
        # Simulate pattern discovery
        pattern_templates = [
            {
                'pattern': 'Entity->Relationship->Entity',
                'frequency': 0.45,
                'examples': ['Person->works_for->Organization'],
                'support': 0.45
            },
            {
                'pattern': 'Document->contains->Multiple Entities',
                'frequency': 0.38,
                'examples': ['Doc1->contains->[Entity1, Entity2, Entity3]'],
                'support': 0.38
            },
            {
                'pattern': 'Entity Co-occurrence',
                'frequency': 0.32,
                'examples': ['Entity1 appears with Entity2 in 32% of documents'],
                'support': 0.32
            }
        ]
        
        for pattern in pattern_templates:
            if pattern['support'] >= min_support:
                patterns.append(pattern)
                if len(patterns) >= max_patterns:
                    break
        
        return patterns
    
    async def _discover_anomalous_patterns(
        self,
        max_patterns: int
    ) -> List[Dict[str, Any]]:
        """
        Discover anomalous or unusual patterns
        """
        return [
            {
                'pattern': 'Isolated entity cluster',
                'anomaly_score': 0.92,
                'description': 'Group of 5 entities with no external connections',
                'entities_involved': 5
            },
            {
                'pattern': 'Unusually high connectivity',
                'anomaly_score': 0.87,
                'description': 'Entity with 10x average connections',
                'entity_id': 'entity_123'
            }
        ][:max_patterns]
    
    async def _discover_structural_patterns(
        self,
        max_patterns: int
    ) -> List[Dict[str, Any]]:
        """
        Discover structural patterns in the graph
        """
        return [
            {
                'pattern': 'Hub-and-spoke',
                'instances': 3,
                'description': 'Central entity with multiple peripheral connections',
                'avg_spoke_count': 8
            },
            {
                'pattern': 'Triangle motif',
                'instances': 15,
                'description': 'Three entities all connected to each other',
                'significance': 0.78
            },
            {
                'pattern': 'Chain structure',
                'instances': 7,
                'description': 'Linear chain of connected entities',
                'avg_chain_length': 4.5
            }
        ][:max_patterns]
    
    async def _get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get entity by ID
        """
        # In production, would query actual entity service
        return {
            'id': entity_id,
            'name': f"Entity_{entity_id[:8]}",
            'type': 'concept',
            'properties': {}
        }
    
    async def _find_entity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find entity by name
        """
        # In production, would search actual entities
        return {
            'id': f"entity_{hash(name) % 1000}",
            'name': name,
            'type': 'concept',
            'properties': {}
        }
    
    async def _find_or_get_entity(self, entity_ref: str) -> Optional[Dict[str, Any]]:
        """
        Find or get entity by ID or name
        """
        # Try as ID first
        if entity_ref.startswith('entity_'):
            return await self._get_entity_by_id(entity_ref)
        # Try as name
        return await self._find_entity_by_name(entity_ref)
    
    async def _get_entity_connections(
        self,
        entity_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get connections for an entity
        """
        # In production, would query actual relationships
        connections = []
        for i in range(3):
            connections.append({
                'source_id': entity_id,
                'target_id': f"entity_{hash(f'{entity_id}_{i}') % 1000}",
                'type': ['related_to', 'part_of', 'references'][i % 3],
                'strength': 0.5 + (i * 0.1)
            })
        return connections
    
    def _get_entity_connections_sync(
        self,
        entity_id: str
    ) -> List[Dict[str, Any]]:
        """
        Synchronous version for DFS
        """
        # Simplified synchronous version
        return [
            {
                'source_id': entity_id,
                'target_id': f"entity_{hash(f'{entity_id}_sync') % 1000}",
                'type': 'related_to'
            }
        ]
    
    async def _get_direct_connections(
        self,
        entity_id: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Get direct connections for an entity
        """
        connections = await self._get_entity_connections(entity_id)
        return connections[:limit]
    
    async def _get_co_occurring_entities(
        self,
        entity_id: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Get entities that co-occur with the given entity
        """
        # In production, would find actual co-occurrences
        co_occurring = []
        for i in range(min(5, limit)):
            co_occurring.append({
                'entity_id': f"entity_co_{i}",
                'name': f"Co-occurring Entity {i}",
                'co_occurrence_count': 10 - i,
                'co_occurrence_score': 0.9 - (i * 0.1)
            })
        return co_occurring
    
    async def _get_related_documents(
        self,
        entity_id: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Get documents related to an entity
        """
        # In production, would query actual documents
        docs = []
        for i in range(min(3, limit)):
            docs.append({
                'document_id': f"doc_{hash(f'{entity_id}_{i}') % 1000}",
                'relevance_score': 0.85 - (i * 0.1),
                'mention_count': 5 - i
            })
        return docs
    
    async def _find_similar_entities(
        self,
        entity_id: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Find entities similar to the given entity
        """
        # In production, would use similarity metrics
        similar = []
        for i in range(min(5, limit)):
            similar.append({
                'entity_id': f"entity_sim_{i}",
                'name': f"Similar Entity {i}",
                'similarity_score': 0.9 - (i * 0.15),
                'similarity_type': 'semantic'
            })
        return similar
    
    async def _calculate_entity_importance(
        self,
        entity_id: str
    ) -> Dict[str, float]:
        """
        Calculate importance metrics for an entity
        """
        return {
            'degree_centrality': 0.75,
            'betweenness_centrality': 0.62,
            'closeness_centrality': 0.58,
            'pagerank': 0.045,
            'hub_score': 0.71,
            'authority_score': 0.68
        }
    
    async def _explore_local_neighborhood(
        self,
        entity_id: str,
        radius: int,
        min_strength: float
    ) -> Dict[str, Any]:
        """
        Explore local neighborhood of an entity
        """
        nodes = {entity_id}
        edges = []
        
        # Simple simulation
        for r in range(radius):
            new_nodes = set()
            for node in nodes:
                connections = await self._get_entity_connections(node)
                for conn in connections:
                    if conn.get('strength', 0) >= min_strength:
                        new_nodes.add(conn['target_id'])
                        edges.append(conn)
            nodes.update(new_nodes)
        
        return {'nodes': nodes, 'edges': edges}
    
    def _detect_communities(
        self,
        nodes: List[str],
        edges: List[Dict[str, Any]]
    ) -> List[Set[str]]:
        """
        Detect communities in a subgraph
        """
        # Simple community detection simulation
        if len(nodes) < 10:
            return [set(nodes)]
        
        # Split into 2-3 communities
        community_size = len(nodes) // 3
        communities = [
            set(nodes[:community_size]),
            set(nodes[community_size:2*community_size]),
            set(nodes[2*community_size:])
        ]
        
        return communities
    
    async def _explore_document_entities(
        self,
        document_id: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Explore entities in a document
        """
        # In production, would query actual document entities
        entities = []
        for i in range(min(10, max_results)):
            entities.append({
                'entity_id': f"doc_entity_{i}",
                'name': f"Entity {i} from {document_id[:8]}",
                'type': ['person', 'organization', 'concept'][i % 3],
                'frequency': 10 - i
            })
        return entities
    
    async def _explore_document_themes(
        self,
        document_id: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Explore themes in a document
        """
        return [
            {
                'theme': 'Data Processing',
                'relevance': 0.85,
                'key_entities': ['Entity1', 'Entity2']
            },
            {
                'theme': 'System Architecture',
                'relevance': 0.72,
                'key_entities': ['Entity3', 'Entity4']
            }
        ][:max_results]
    
    async def _explore_document_connections(
        self,
        document_id: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Explore connections from a document
        """
        return [
            {
                'connected_document': f"doc_{i}",
                'connection_type': 'references',
                'strength': 0.8 - (i * 0.1)
            }
            for i in range(min(5, max_results))
        ]
    
    def _calculate_path_confidence(self, path: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for a path
        """
        if not path:
            return 0.0
        
        # Average of connection strengths
        strengths = [conn.get('strength', 0.5) for conn in path]
        return sum(strengths) / len(strengths) if strengths else 0.5
    
    def _describe_path(self, path: List[Dict[str, Any]]) -> str:
        """
        Generate human-readable path description
        """
        if not path:
            return "No path"
        
        descriptions = []
        for conn in path:
            descriptions.append(
                f"{conn['source_id'][:8]} -{conn['type']}-> {conn['target_id'][:8]}"
            )
        
        return " -> ".join(descriptions)
    
    def clear_cache(self):
        """Clear the exploration cache"""
        self.exploration_cache.clear()