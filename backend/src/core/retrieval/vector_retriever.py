"""
Vector-based retrieval using Qdrant with enhanced ID mapping support
"""
from typing import List, Dict, Any, Optional
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from ...models import RetrievalResult, Chunk
from ...services.id_mapper import IDMapper
from ...db import get_session

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Handles vector-based retrieval from Qdrant with ID mapping integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        vector_config = config.get('vector_store', {})
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            host=vector_config.get('host', 'localhost'),
            port=vector_config.get('port', 6333)
        )
        
        self.collection_name = vector_config.get('collection_name', 'rag_chunks')
        self.embedding_dim = vector_config.get('embedding_dim', 1536)
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
    
    def index_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """Index chunks with their embeddings and ID mappings"""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        points = []
        
        with get_session() as session:
            mapper = IDMapper(session)
            
            for chunk, embedding in zip(chunks, embeddings):
                # Get related IDs from mapper
                entity_ids = chunk.entity_ids if hasattr(chunk, 'entity_ids') else []
                graph_node_ids = chunk.graph_node_ids if hasattr(chunk, 'graph_node_ids') else []
                
                # Create vector ID if not exists
                vector_id = chunk.vector_id if hasattr(chunk, 'vector_id') else chunk.id
                
                # Create ID mappings
                mapper.add_link("chunk", chunk.id, "vector", vector_id, relation="embeds")
                
                # Add entity links
                for entity_id in entity_ids:
                    mapper.add_link("chunk", chunk.id, "entity", entity_id, relation="mentions")
                
                # Add graph node links
                for node_id in graph_node_ids:
                    mapper.add_link("chunk", chunk.id, "graph_node", node_id, relation="represents")
                
                point = PointStruct(
                    id=vector_id,
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.id,
                        "content": chunk.content,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "chunk_type": chunk.chunk_type,
                        "tokens": chunk.tokens,
                        "metadata": chunk.metadata,
                        "entity_ids": entity_ids,
                        "graph_node_ids": graph_node_ids,
                        "parent_id": chunk.parent_id if hasattr(chunk, 'parent_id') else None,
                        "children_ids": chunk.children_ids if hasattr(chunk, 'children_ids') else []
                    }
                )
                points.append(point)
        
        # Batch upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Indexed {len(points)} chunks to Qdrant with ID mappings")
    
    def retrieve(self, query_embedding: List[float], 
                k: int = 10, 
                filter_dict: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks based on vector similarity
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_dict: Optional filters for metadata
        
        Returns:
            List of retrieval results with ID mapping information
        """
        # Build filter if provided
        qdrant_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)
        
        # Search Qdrant
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=k,
                with_payload=True
            )
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return []
        
        # Convert to RetrievalResult
        results = []
        for hit in search_results:
            payload = hit.payload
            
            result = RetrievalResult(
                chunk_id=payload.get("chunk_id", str(hit.id)),
                content=payload.get("content", ""),
                score=hit.score,
                source="vector",
                metadata={
                    **payload.get("metadata", {}),
                    "document_id": payload.get("document_id"),
                    "chunk_type": payload.get("chunk_type"),
                    "chunk_index": payload.get("chunk_index"),
                    "entity_ids": payload.get("entity_ids", []),
                    "graph_node_ids": payload.get("graph_node_ids", []),
                    "parent_id": payload.get("parent_id"),
                    "children_ids": payload.get("children_ids", [])
                },
                highlights=[]
            )
            results.append(result)
        
        return results
    
    def retrieve_with_graph_seeds(self, query_embedding: List[float], k: int = 10) -> Dict[str, Any]:
        """
        Retrieve chunks and extract seed nodes for graph traversal
        
        Args:
            query_embedding: Query vector
            k: Number of results
            
        Returns:
            Dictionary with chunks and seed nodes
        """
        results = self.retrieve(query_embedding, k)
        
        # Extract unique entity and graph node IDs as seeds
        entity_seeds = set()
        graph_seeds = set()
        
        for result in results:
            entity_ids = result.metadata.get("entity_ids", [])
            graph_node_ids = result.metadata.get("graph_node_ids", [])
            
            entity_seeds.update(entity_ids)
            graph_seeds.update(graph_node_ids)
        
        return {
            "chunks": results,
            "entity_seeds": list(entity_seeds),
            "graph_seeds": list(graph_seeds),
            "total_seeds": len(entity_seeds) + len(graph_seeds)
        }
    
    def delete_chunks(self, chunk_ids: List[str]):
        """Delete chunks from vector store and update ID mappings"""
        with get_session() as session:
            mapper = IDMapper(session)
            
            vector_ids = []
            for chunk_id in chunk_ids:
                vector_id = mapper.get_vector_for_chunk(chunk_id)
                if vector_id:
                    vector_ids.append(vector_id)
        
        if vector_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=vector_ids
            )
            logger.info(f"Deleted {len(vector_ids)} vectors from Qdrant")
    
    def update_chunk_embeddings(self, chunk_id: str, new_embedding: List[float]):
        """Update embedding for a specific chunk"""
        with get_session() as session:
            mapper = IDMapper(session)
            vector_id = mapper.get_vector_for_chunk(chunk_id)
            
            if vector_id:
                # Get existing payload
                results = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[vector_id],
                    with_payload=True
                )
                
                if results:
                    existing_payload = results[0].payload
                    
                    # Update vector
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=[
                            PointStruct(
                                id=vector_id,
                                vector=new_embedding,
                                payload=existing_payload
                            )
                        ]
                    )
                    logger.info(f"Updated embedding for chunk {chunk_id}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}