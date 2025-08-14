"""
Embedding service for generating vector embeddings
"""
import os
from typing import List, Optional
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
import hashlib

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using various models"""
    
    def __init__(self):
        self.model = None
        self.openai_client = None
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize or reinitialize services with current environment variables"""
        # Force reload .env file and read fresh
        from dotenv import load_dotenv
        load_dotenv(override=True)
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers")
        
        print(f"🔑 EmbeddingService loading API key: {self.openai_api_key[:20]}...{self.openai_api_key[-4:]}")
        
        if self.embedding_model == "openai" and self.openai_api_key and OPENAI_AVAILABLE:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            self.model = None
            self.model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            logger.info("Using OpenAI embeddings")
        else:
            # Use sentence-transformers as default
            try:
                model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
                self.model = SentenceTransformer(model_name)
                self.model_name = model_name
                self.embedding_model = "sentence-transformers"
                logger.info(f"Using Sentence Transformers: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.model = None
                self.embedding_model = "mock"
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not text:
            raise ValueError("Cannot generate embedding for empty text")
        
        # Always reinitialize services to pick up latest environment variables
        self._initialize_services()
        
        try:
            if self.embedding_model == "openai" and self.openai_client:
                # Use OpenAI API with proper v1.x syntax
                response = self.openai_client.embeddings.create(
                    model=self.model_name,
                    input=text
                )
                return response.data[0].embedding
            
            elif self.embedding_model == "sentence-transformers" and self.model:
                # Use sentence-transformers
                embedding = self.model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            
            else:
                # No fallback - must use real embeddings
                raise RuntimeError(f"Embedding service not properly configured: model={self.embedding_model}")
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Try to reinitialize once if API key error
            if "401" in str(e) or "Incorrect API key" in str(e):
                logger.info("API key error detected, reinitializing services...")
                self._initialize_services()
            raise  # Don't fall back to mock
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        
        # Always reinitialize services to pick up latest environment variables
        self._initialize_services()
        
        try:
            if self.embedding_model == "openai" and self.openai_client:
                # Batch process with OpenAI v1.x API
                response = self.openai_client.embeddings.create(
                    model=self.model_name,
                    input=texts
                )
                return [item.embedding for item in response.data]
            
            elif self.embedding_model == "sentence-transformers" and self.model:
                # Batch process with sentence-transformers
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                return embeddings.tolist()
            
            else:
                # No fallback - must use real embeddings
                raise RuntimeError(f"Embedding service not properly configured: model={self.embedding_model}")
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Try to reinitialize once if API key error
            if "401" in str(e) or "Incorrect API key" in str(e):
                logger.info("API key error detected, reinitializing services...")
                self._initialize_services()
            raise  # Don't fall back to mock
    
    def _mock_embedding(self, text: str) -> List[float]:
        """Generate a deterministic mock embedding based on text hash"""
        # Create a deterministic embedding based on text content
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to numbers
        np.random.seed(int(text_hash[:8], 16))
        
        # Generate embedding of appropriate size
        if self.embedding_model == "openai":
            dim = 1536  # OpenAI ada-002 dimension
        else:
            dim = 768  # Default sentence-transformer dimension
        
        # Generate random but deterministic embedding
        embedding = np.random.randn(dim) * 0.1
        
        # Add some text-based features
        text_lower = text.lower()
        
        # Boost certain dimensions based on content
        if "redis" in text_lower:
            embedding[0] += 0.5
        if "cache" in text_lower:
            embedding[1] += 0.5
        if "memory" in text_lower:
            embedding[2] += 0.5
        if "database" in text_lower:
            embedding[3] += 0.5
        if "graph" in text_lower:
            embedding[4] += 0.5
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by current model"""
        if self.embedding_model == "openai":
            return 1536
        elif self.model:
            # Get dimension from sentence-transformer model
            return self.model.get_sentence_embedding_dimension()
        else:
            return 768  # Default dimension


# Dependency injection - create fresh instance each time for config changes
def get_embedding_service():
    """Get a fresh embedding service instance"""
    return EmbeddingService()