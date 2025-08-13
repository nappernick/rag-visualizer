"""
Vector storage service using Qdrant
"""
import os
from typing import List, Dict, Optional, Any
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    UpdateStatus
)
import numpy as np
from uuid import uuid4

logger = logging.getLogger(__name__)


class VectorService:
    """Service for managing vector storage in Qdrant"""
    
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL", "")
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        
        try:
            if qdrant_url and qdrant_api_key:
                self.client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key
                )
            elif qdrant_url:
                self.client = QdrantClient(url=qdrant_url)
            else:
                # Fallback to localhost
                self.client = QdrantClient(host="localhost", port=6333)
            
            self.initialized = True
            self.collection_name = "rag_chunks"
            self.vector_size = 768  # Default for sentence-transformers
            
            # Ensure collection exists
            self._ensure_collection()
            logger.info("Qdrant client initialized successfully")
            
        except Exception as e:
            self.client = None
            self.initialized = False
            logger.warning(f"Qdrant initialization failed: {e}")
    
    def _ensure_collection(self):
        """Ensure the collection exists in Qdrant"""
        if not self.initialized:
            return
        
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
    
    async def store_vectors(self, chunks: List[Dict], embeddings: List[List[float]]) -> bool:
        """Store chunk vectors in Qdrant"""
        if not self.initialized or not chunks or not embeddings:
            return False
        
        if len(chunks) != len(embeddings):
            logger.error("Chunks and embeddings length mismatch")
            return False
        
        try:
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                point_id = str(uuid4())
                
                # Store chunk metadata as payload
                payload = {
                    "chunk_id": chunk["id"],
                    "document_id": chunk["document_id"],
                    "content": chunk["content"],
                    "chunk_index": chunk.get("chunk_index", 0),
                    "chunk_type": chunk.get("chunk_type", "standard"),
                    "created_at": chunk.get("created_at", ""),
                    "doc_type": chunk.get("metadata", {}).get("doc_type", "default")
                }
                
                # Add temporal metadata if available
                if "metadata" in chunk:
                    if "temporal_score" in chunk["metadata"]:
                        payload["temporal_score"] = chunk["metadata"]["temporal_score"]
                    if "created_at_ms" in chunk["metadata"]:
                        payload["created_at_ms"] = chunk["metadata"]["created_at_ms"]
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                ))
            
            # Upload points to Qdrant
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points
            )
            
            if operation_info.status == UpdateStatus.COMPLETED:
                logger.info(f"Stored {len(points)} vectors in Qdrant")
                return True
            else:
                logger.error(f"Failed to store vectors: {operation_info}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing vectors: {e}")
            return False
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        document_id: Optional[str] = None,
        doc_type: Optional[str] = None,
        score_threshold: float = 0.0
    ) -> List[Dict]:
        """Search for similar chunks in Qdrant"""
        if not self.initialized:
            return []
        
        try:
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
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for hit in search_result:
                result = {
                    "chunk_id": hit.payload.get("chunk_id"),
                    "document_id": hit.payload.get("document_id"),
                    "content": hit.payload.get("content"),
                    "score": hit.score,
                    "chunk_index": hit.payload.get("chunk_index", 0),
                    "chunk_type": hit.payload.get("chunk_type", "standard"),
                    "temporal_score": hit.payload.get("temporal_score", 1.0),
                    "metadata": {
                        "doc_type": hit.payload.get("doc_type", "default"),
                        "created_at": hit.payload.get("created_at", "")
                    }
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []
    
    async def get_chunk_vector(self, chunk_id: str) -> Optional[List[float]]:
        """Retrieve vector for a specific chunk"""
        if not self.initialized:
            return None
        
        try:
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
            
            if result[0]:
                return result[0][0].vector
            return None
            
        except Exception as e:
            logger.error(f"Error getting chunk vector: {e}")
            return None
    
    async def delete_document_vectors(self, document_id: str) -> bool:
        """Delete all vectors for a document"""
        if not self.initialized:
            return False
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
            logger.info(f"Deleted vectors for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document vectors: {e}")
            return False
    
    async def clear_all(self) -> bool:
        """Clear all vectors from the collection"""
        if not self.initialized:
            return False
        
        try:
            # Delete and recreate collection
            self.client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()
            logger.info("Cleared all vectors from Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing vectors: {e}")
            return False


# Global vector service instance
vector_service = VectorService()