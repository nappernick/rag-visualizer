"""
Qdrant Connection Manager with Retry Logic and Health Checks

Provides robust connection management for Qdrant with:
- Automatic retry logic with exponential backoff
- Connection health checks
- Proper timeout configuration
- Graceful error handling
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List, Callable
from functools import wraps
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    UpdateStatus
)
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

logger = logging.getLogger(__name__)


class QdrantConnectionManager:
    """Manages Qdrant connections with automatic retry and health checks."""
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "rag_chunks",
        vector_size: int = 1536,  # Updated for OpenAI text-embedding-3-small
        max_retry_attempts: int = 3,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 10.0,
        timeout: int = 60
    ):
        """
        Initialize the connection manager.
        
        Args:
            url: Qdrant cloud URL
            api_key: API key for authentication
            collection_name: Name of the collection to use
            vector_size: Dimension of vectors (1536 for text-embedding-3-small)
            max_retry_attempts: Maximum number of retry attempts
            initial_retry_delay: Initial delay between retries in seconds
            max_retry_delay: Maximum delay between retries in seconds
            timeout: Request timeout in seconds
        """
        # Load from environment if not provided
        self.url = url or os.getenv("QDRANT_URL", "")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY", "")
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "rag_chunks")
        self.vector_size = vector_size or int(os.getenv("EMBED_DIM", "1536"))
        
        self.max_retry_attempts = max_retry_attempts
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        self.timeout = timeout
        
        self._client = None
        self.initialized = False
        
        # Try to connect
        self._connect()
    
    def _connect(self) -> bool:
        """Establish connection to Qdrant with retry logic."""
        if not self.url or not self.api_key:
            logger.error("Qdrant URL or API key not configured")
            return False
        
        last_error = None
        retry_delay = self.initial_retry_delay
        
        for attempt in range(self.max_retry_attempts):
            try:
                logger.info(f"Attempting to connect to Qdrant (attempt {attempt + 1}/{self.max_retry_attempts})")
                
                # Create client with proper timeout
                self._client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    prefer_grpc=False  # Use HTTP for better compatibility
                )
                
                # Test connection by getting collections
                collections = self._client.get_collections()
                logger.info(f"Successfully connected to Qdrant. Found {len(collections.collections)} collections")
                
                # Ensure our collection exists
                self._ensure_collection()
                
                self.initialized = True
                return True
                
            except (ResponseHandlingException, UnexpectedResponse, TimeoutError) as e:
                last_error = e
                logger.warning(f"Qdrant connection attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retry_attempts - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, self.max_retry_delay)
                    
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error connecting to Qdrant: {e}")
                if attempt < self.max_retry_attempts - 1:
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, self.max_retry_delay)
        
        logger.error(f"Failed to connect to Qdrant after {self.max_retry_attempts} attempts: {last_error}")
        self.initialized = False
        return False
    
    def _ensure_collection(self) -> bool:
        """Ensure the collection exists in Qdrant."""
        if not self._client:
            return False
        
        try:
            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection '{self.collection_name}' with vector size {self.vector_size}")
                
                self._client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                
                # Create payload index for better search performance
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_id",
                    field_schema="keyword"
                )
                
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="chunk_id",
                    field_schema="keyword"
                )
                
                logger.info(f"Created collection '{self.collection_name}' with indexes")
            else:
                # Verify collection has correct vector size
                collection_info = self._client.get_collection(self.collection_name)
                current_size = collection_info.config.params.vectors.size
                
                if current_size != self.vector_size:
                    logger.warning(f"Collection vector size mismatch: expected {self.vector_size}, got {current_size}")
                    logger.info(f"Recreating collection with correct vector size")
                    
                    self._client.delete_collection(self.collection_name)
                    self._client.recreate_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.vector_size,
                            distance=Distance.COSINE
                        )
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            return False
    
    def health_check(self) -> bool:
        """
        Check if the Qdrant connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        if not self._client:
            return False
        
        try:
            # Try to get collection info
            collection_info = self._client.get_collection(self.collection_name)
            return collection_info is not None
            
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    def ensure_connected(self) -> bool:
        """Ensure connection is established, reconnecting if necessary."""
        if not self.initialized or not self.health_check():
            logger.info("Connection unhealthy, attempting to reconnect...")
            return self._connect()
        return True
    
    def with_retry(self, func: Callable) -> Callable:
        """
        Decorator to add retry logic to Qdrant operations.
        
        Args:
            func: Function to wrap with retry logic
            
        Returns:
            Wrapped function with retry capability
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            retry_delay = self.initial_retry_delay
            
            for attempt in range(self.max_retry_attempts):
                try:
                    # Ensure connection before operation
                    if not self.ensure_connected():
                        raise ConnectionError("Failed to connect to Qdrant")
                    
                    return func(*args, **kwargs)
                    
                except (ResponseHandlingException, UnexpectedResponse, TimeoutError) as e:
                    last_error = e
                    logger.warning(f"Transient error in {func.__name__} (attempt {attempt + 1}): {e}")
                    
                    if attempt < self.max_retry_attempts - 1:
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, self.max_retry_delay)
                        
                except Exception as e:
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise
            
            raise last_error
        
        return wrapper
    
    def get_client(self) -> Optional[QdrantClient]:
        """Get the Qdrant client, ensuring it's connected."""
        if self.ensure_connected():
            return self._client
        return None
    
    @property
    def client(self) -> Optional[QdrantClient]:
        """Property to get the client."""
        return self.get_client()
    
    def close(self):
        """Close the Qdrant client connection."""
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")
            finally:
                self._client = None
                self.initialized = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Cleanup on garbage collection."""
        self.close()


# Global connection manager instance
_connection_manager = None


def get_qdrant_connection_manager() -> QdrantConnectionManager:
    """Get or create the global Qdrant connection manager."""
    global _connection_manager
    
    if _connection_manager is None:
        _connection_manager = QdrantConnectionManager()
    
    return _connection_manager


def reset_qdrant_connection():
    """Reset the global Qdrant connection."""
    global _connection_manager
    
    if _connection_manager:
        _connection_manager.close()
        _connection_manager = None