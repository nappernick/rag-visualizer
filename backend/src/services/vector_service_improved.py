"""
Improved Vector storage service using Qdrant with robust connection management
"""
import os
from typing import List, Dict, Optional, Any
import logging
from uuid import uuid4
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    UpdateStatus
)
import numpy as np

from .qdrant_connection_manager import get_qdrant_connection_manager

logger = logging.getLogger(__name__)


class ImprovedVectorService:
    """Service for managing vector storage in Qdrant with improved reliability"""
    
    def __init__(self):
        """Initialize the vector service with connection manager."""
        self.conn_manager = get_qdrant_connection_manager()
        self.collection_name = self.conn_manager.collection_name
        self.vector_size = self.conn_manager.vector_size
        self.initialized = self.conn_manager.initialized
        
        if self.initialized:
            logger.info(f"Vector service initialized with collection '{self.collection_name}'")
        else:
            logger.warning("Vector service initialization failed - will retry on operations")
    
    @property
    def client(self):
        """Get the Qdrant client from connection manager."""
        return self.conn_manager.client
    
    def _ensure_initialized(self) -> bool:
        """Ensure the service is initialized before operations."""
        if not self.initialized:
            self.initialized = self.conn_manager.ensure_connected()
        return self.initialized
    
    async def store_vectors(self, chunks: List[Dict], embeddings: List[List[float]]) -> bool:
        """
        Store chunk vectors in Qdrant with retry logic.
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: List of embedding vectors
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_initialized():
            logger.error("Cannot store vectors - service not initialized")
            return False
        
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings to store")
            return False
        
        if len(chunks) != len(embeddings):
            logger.error(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch")
            return False
        
        @self.conn_manager.with_retry
        def _store():
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                # Generate unique ID for the point
                point_id = str(uuid4())
                
                # Prepare payload with all chunk metadata
                payload = {
                    "chunk_id": chunk.get("id", str(uuid4())),
                    "document_id": chunk.get("document_id", ""),
                    "content": chunk.get("content", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "chunk_type": chunk.get("chunk_type", "standard"),
                    "position": chunk.get("position", 0),
                    "created_at": chunk.get("created_at", ""),
                }
                
                # Add document type from metadata
                metadata = chunk.get("metadata", {})
                if metadata:
                    payload["doc_type"] = metadata.get("doc_type", "default")
                    
                    # Add temporal metadata if available
                    if "temporal_score" in metadata:
                        payload["temporal_score"] = metadata["temporal_score"]
                    if "created_at_ms" in metadata:
                        payload["created_at_ms"] = metadata["created_at_ms"]
                    if "document_date" in metadata:
                        payload["document_date"] = metadata["document_date"]
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                ))
            
            # Upload points to Qdrant with wait for completion
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points
            )
            
            if operation_info.status == UpdateStatus.COMPLETED:
                logger.info(f"Successfully stored {len(points)} vectors in Qdrant")
                return True
            else:
                logger.error(f"Failed to store vectors: {operation_info}")
                return False
        
        try:
            return _store()
        except Exception as e:
            logger.error(f"Error storing vectors after retries: {e}")
            return False
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        document_id: Optional[str] = None,
        doc_type: Optional[str] = None,
        score_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar chunks in Qdrant with retry logic.
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            document_id: Filter by document ID
            doc_type: Filter by document type
            score_threshold: Minimum similarity score
            
        Returns:
            List of similar chunks with scores
        """
        if not self._ensure_initialized():
            logger.error("Cannot search - service not initialized")
            return []
        
        @self.conn_manager.with_retry
        def _search():
            # Build filter conditions
            must_conditions = []
            
            if document_id:
                must_conditions.append(
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                )
            
            if doc_type:
                must_conditions.append(
                    FieldCondition(
                        key="doc_type",
                        match=MatchValue(value=doc_type)
                    )
                )
            
            # Perform search
            search_filter = Filter(must=must_conditions) if must_conditions else None
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=search_filter,
                score_threshold=score_threshold if score_threshold > 0 else None,
                with_payload=True
            )
            
            # Format results
            results = []
            for hit in search_result:
                result = {
                    "chunk_id": hit.payload.get("chunk_id"),
                    "document_id": hit.payload.get("document_id"),
                    "content": hit.payload.get("content"),
                    "score": float(hit.score),
                    "chunk_index": hit.payload.get("chunk_index", 0),
                    "position": hit.payload.get("position", 0),
                    "chunk_type": hit.payload.get("chunk_type", "standard"),
                    "temporal_score": hit.payload.get("temporal_score", 1.0),
                    "metadata": {
                        "doc_type": hit.payload.get("doc_type", "default"),
                        "created_at": hit.payload.get("created_at", ""),
                        "document_date": hit.payload.get("document_date", "")
                    }
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks")
            return results
        
        try:
            return _search()
        except Exception as e:
            logger.error(f"Error searching similar chunks after retries: {e}")
            return []
    
    async def get_chunk_vector(self, chunk_id: str) -> Optional[List[float]]:
        """
        Retrieve vector for a specific chunk with retry logic.
        
        Args:
            chunk_id: ID of the chunk
            
        Returns:
            Vector if found, None otherwise
        """
        if not self._ensure_initialized():
            return None
        
        @self.conn_manager.with_retry
        def _get_vector():
            # Search by chunk_id in payload
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="chunk_id",
                            match=MatchValue(value=chunk_id)
                        )
                    ]
                ),
                limit=1,
                with_vectors=True
            )
            
            if result[0]:  # Check if we got any results
                return result[0][0].vector
            return None
        
        try:
            return _get_vector()
        except Exception as e:
            logger.error(f"Error getting chunk vector after retries: {e}")
            return None
    
    async def delete_document_vectors(self, document_id: str) -> bool:
        """
        Delete all vectors for a document with retry logic.
        
        Args:
            document_id: Document ID to delete vectors for
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_initialized():
            return False
        
        @self.conn_manager.with_retry
        def _delete():
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                wait=True
            )
            
            logger.info(f"Deleted vectors for document {document_id}")
            return True
        
        try:
            return _delete()
        except Exception as e:
            logger.error(f"Error deleting document vectors after retries: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self._ensure_initialized():
            return {"error": "Service not initialized"}
        
        @self.conn_manager.with_retry
        def _get_stats():
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vector_size": collection_info.config.params.vectors.size,
                "points_count": collection_info.points_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status.status if collection_info.optimizer_status else "unknown"
            }
        
        try:
            return _get_stats()
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    async def clear_all(self) -> bool:
        """
        Clear all vectors from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_initialized():
            return False
        
        @self.conn_manager.with_retry
        def _clear():
            # Delete and recreate collection
            self.client.delete_collection(collection_name=self.collection_name)
            self.conn_manager._ensure_collection()
            logger.info("Cleared all vectors from Qdrant")
            return True
        
        try:
            return _clear()
        except Exception as e:
            logger.error(f"Error clearing vectors after retries: {e}")
            return False
    
    def close(self):
        """Close the connection manager."""
        if self.conn_manager:
            self.conn_manager.close()


# Create a global instance for backwards compatibility
improved_vector_service = ImprovedVectorService()