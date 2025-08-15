"""
Graph Traversal Service - Handles graph traversal and pathfinding operations
"""
from typing import List, Dict, Any, Optional, Set, Tuple
import logging
from collections import deque, defaultdict
import heapq

logger = logging.getLogger(__name__)


class GraphTraversalService:
    """Service for graph traversal and pathfinding operations"""
    
    def __init__(self, graph_store=None):
        self.graph_store = graph_store
        self.path_cache = {}
    
    async def find_shortest_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        relationship_types: Optional[List[str]] = None
    ) -> Optional[List[Tuple[str, str, str]]]:
        """
        Find the shortest path between two entities
        
        Args:
            start_id: Starting entity ID
            end_id: Target entity ID
            max_depth: Maximum path length
            relationship_types: Optional relationship type filter
            
        Returns:
            Shortest path as list of (source, relationship, target) tuples
        """
        cache_key = f"{start_id}->{end_id}"
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        if not self.graph_store:
            return None
        
        # Build relationship filter
        rel_filter = ""
        if relationship_types:
            rel_filter = f"[:{':'.join(relationship_types)}]"
        
        query = f"""
        MATCH path = shortestPath((start)-{rel_filter}*..{max_depth}-(end))
        WHERE start.id = '{start_id}' AND end.id = '{end_id}'
        RETURN path
        """
        
        try:
            results = await self.graph_store.execute_query(query)
            
            if results:
                path = self._extract_path(results[0]['path'])
                self.path_cache[cache_key] = path
                return path
        except Exception as e:
            logger.error(f"Error finding shortest path: {e}")
        
        return None
    
    async def find_all_paths(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 3,
        max_paths: int = 10,
        relationship_types: Optional[List[str]] = None
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Find all paths between two entities up to max_depth
        
        Args:
            start_id: Starting entity ID
            end_id: Target entity ID
            max_depth: Maximum path length
            max_paths: Maximum number of paths to return
            relationship_types: Optional relationship type filter
            
        Returns:
            List of paths, each path is list of (source, relationship, target) tuples
        """
        if not self.graph_store:
            return []
        
        rel_filter = ""
        if relationship_types:
            rel_filter = f"[:{':'.join(relationship_types)}]"
        
        query = f"""
        MATCH path = (start)-{rel_filter}*1..{max_depth}-(end)
        WHERE start.id = '{start_id}' AND end.id = '{end_id}'
        RETURN path
        LIMIT {max_paths}
        """
        
        paths = []
        try:
            results = await self.graph_store.execute_query(query)
            
            for result in results:
                path = self._extract_path(result['path'])
                paths.append(path)
        except Exception as e:
            logger.error(f"Error finding all paths: {e}")
        
        return paths
    
    async def traverse_breadth_first(
        self,
        start_id: str,
        max_depth: int = 2,
        relationship_types: Optional[List[str]] = None,
        max_nodes: int = 100
    ) -> Dict[str, Any]:
        """
        Perform breadth-first traversal from a starting node
        
        Args:
            start_id: Starting entity ID
            max_depth: Maximum traversal depth
            relationship_types: Optional relationship type filter
            max_nodes: Maximum number of nodes to visit
            
        Returns:
            Traversal result with visited nodes and relationships
        """
        visited_nodes = set()
        visited_edges = set()
        node_levels = {start_id: 0}
        queue = deque([(start_id, 0)])
        
        nodes = []
        edges = []
        
        while queue and len(visited_nodes) < max_nodes:
            current_id, depth = queue.popleft()
            
            if current_id in visited_nodes or depth > max_depth:
                continue
            
            visited_nodes.add(current_id)
            
            # Get node details
            node = await self._get_node_details(current_id)
            if node:
                nodes.append(node)
            
            # Get neighbors
            if depth < max_depth:
                neighbors = await self._get_neighbors(
                    current_id,
                    relationship_types
                )
                
                for neighbor_id, edge_info in neighbors:
                    edge_key = (current_id, neighbor_id, edge_info['type'])
                    
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        edges.append(edge_info)
                    
                    if neighbor_id not in visited_nodes:
                        queue.append((neighbor_id, depth + 1))
                        node_levels[neighbor_id] = depth + 1
        
        return {
            'nodes': nodes,
            'edges': edges,
            'node_levels': node_levels,
            'traversal_type': 'breadth_first',
            'max_depth': max_depth,
            'nodes_visited': len(visited_nodes)
        }
    
    async def find_connected_components(
        self,
        entity_ids: Optional[List[str]] = None,
        min_component_size: int = 2
    ) -> List[Set[str]]:
        """
        Find connected components in the graph
        
        Args:
            entity_ids: Optional list of entity IDs to consider
            min_component_size: Minimum size for a component to be included
            
        Returns:
            List of connected components (sets of entity IDs)
        """
        if not self.graph_store:
            return []
        
        components = []
        visited = set()
        
        # Get all nodes if not specified
        if not entity_ids:
            query = "MATCH (n) RETURN n.id as id"
            results = await self.graph_store.execute_query(query)
            entity_ids = [r['id'] for r in results]
        
        for entity_id in entity_ids:
            if entity_id not in visited:
                component = await self._explore_component(entity_id, visited)
                if len(component) >= min_component_size:
                    components.append(component)
        
        return components
    
    async def calculate_centrality(
        self,
        centrality_type: str = "degree",
        entity_ids: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate centrality metrics for nodes
        
        Args:
            centrality_type: Type of centrality (degree, betweenness, closeness)
            entity_ids: Optional list of entity IDs to calculate for
            
        Returns:
            Dictionary mapping entity IDs to centrality scores
        """
        if not self.graph_store:
            return {}
        
        if centrality_type == "degree":
            return await self._calculate_degree_centrality(entity_ids)
        elif centrality_type == "betweenness":
            return await self._calculate_betweenness_centrality(entity_ids)
        elif centrality_type == "closeness":
            return await self._calculate_closeness_centrality(entity_ids)
        else:
            logger.warning(f"Unknown centrality type: {centrality_type}")
            return {}
    
    async def find_communities(
        self,
        algorithm: str = "louvain",
        resolution: float = 1.0
    ) -> List[Set[str]]:
        """
        Detect communities in the graph
        
        Args:
            algorithm: Community detection algorithm
            resolution: Resolution parameter for some algorithms
            
        Returns:
            List of communities (sets of entity IDs)
        """
        if not self.graph_store:
            return []
        
        # Simplified community detection using connected components
        # In production, would use proper algorithms like Louvain
        return await self.find_connected_components(min_component_size=3)
    
    async def get_subgraph(
        self,
        entity_ids: List[str],
        include_edges: bool = True,
        expand_depth: int = 0
    ) -> Dict[str, Any]:
        """
        Extract a subgraph containing specified entities
        
        Args:
            entity_ids: List of entity IDs to include
            include_edges: Whether to include edges between entities
            expand_depth: How many hops to expand from specified entities
            
        Returns:
            Subgraph with nodes and edges
        """
        nodes = []
        edges = []
        expanded_ids = set(entity_ids)
        
        # Expand if requested
        if expand_depth > 0:
            for entity_id in entity_ids:
                traversal = await self.traverse_breadth_first(
                    entity_id,
                    max_depth=expand_depth,
                    max_nodes=50
                )
                expanded_ids.update(
                    node['id'] for node in traversal['nodes']
                )
        
        # Get nodes
        for entity_id in expanded_ids:
            node = await self._get_node_details(entity_id)
            if node:
                nodes.append(node)
        
        # Get edges if requested
        if include_edges and len(expanded_ids) > 1:
            query = f"""
            MATCH (n)-[r]->(m)
            WHERE n.id IN {list(expanded_ids)}
            AND m.id IN {list(expanded_ids)}
            RETURN n.id as source, m.id as target, r
            """
            
            results = await self.graph_store.execute_query(query)
            
            for result in results:
                edges.append({
                    'source': result['source'],
                    'target': result['target'],
                    'type': result['r']['type'],
                    'properties': result['r'].get('properties', {})
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'node_count': len(nodes),
            'edge_count': len(edges)
        }
    
    # Helper methods
    def _extract_path(self, path_data: Dict) -> List[Tuple[str, str, str]]:
        """Extract path as list of (source, relationship, target) tuples"""
        path = []
        nodes = path_data.get('nodes', [])
        relationships = path_data.get('relationships', [])
        
        for i in range(len(relationships)):
            path.append((
                nodes[i].get('id'),
                relationships[i].get('type'),
                nodes[i + 1].get('id')
            ))
        
        return path
    
    async def _get_node_details(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a node"""
        if not self.graph_store:
            return None
        
        query = f"MATCH (n) WHERE n.id = '{node_id}' RETURN n"
        results = await self.graph_store.execute_query(query)
        
        if results:
            node = results[0]['n']
            return {
                'id': node_id,
                'labels': node.get('labels', []),
                'properties': node.get('properties', {})
            }
        
        return None
    
    async def _get_neighbors(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Get neighboring nodes and connecting edges"""
        if not self.graph_store:
            return []
        
        query = f"MATCH (n)-[r]-(m) WHERE n.id = '{node_id}'"
        
        if relationship_types:
            query += f" AND type(r) IN {relationship_types}"
        
        query += " RETURN m.id as neighbor_id, r, startNode(r).id as start"
        
        results = await self.graph_store.execute_query(query)
        neighbors = []
        
        for result in results:
            edge_info = {
                'source': node_id if result['start'] == node_id else result['neighbor_id'],
                'target': result['neighbor_id'] if result['start'] == node_id else node_id,
                'type': result['r']['type'],
                'properties': result['r'].get('properties', {})
            }
            neighbors.append((result['neighbor_id'], edge_info))
        
        return neighbors
    
    async def _explore_component(
        self,
        start_id: str,
        visited: Set[str]
    ) -> Set[str]:
        """Explore a connected component using DFS"""
        component = set()
        stack = [start_id]
        
        while stack:
            current_id = stack.pop()
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            component.add(current_id)
            
            neighbors = await self._get_neighbors(current_id)
            for neighbor_id, _ in neighbors:
                if neighbor_id not in visited:
                    stack.append(neighbor_id)
        
        return component
    
    async def _calculate_degree_centrality(
        self,
        entity_ids: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Calculate degree centrality for nodes"""
        if not self.graph_store:
            return {}
        
        query = "MATCH (n)-[r]-() "
        if entity_ids:
            query += f"WHERE n.id IN {entity_ids} "
        query += "RETURN n.id as id, count(r) as degree"
        
        results = await self.graph_store.execute_query(query)
        
        centrality = {}
        max_degree = max((r['degree'] for r in results), default=1)
        
        for result in results:
            centrality[result['id']] = result['degree'] / max_degree
        
        return centrality
    
    async def _calculate_betweenness_centrality(
        self,
        entity_ids: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Calculate betweenness centrality (simplified)"""
        # Simplified implementation - in production would use proper algorithm
        return await self._calculate_degree_centrality(entity_ids)
    
    async def _calculate_closeness_centrality(
        self,
        entity_ids: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Calculate closeness centrality (simplified)"""
        # Simplified implementation - in production would use proper algorithm
        return await self._calculate_degree_centrality(entity_ids)
    
    def clear_cache(self):
        """Clear the path cache"""
        self.path_cache.clear()