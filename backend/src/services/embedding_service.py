"""
Embedding service for generating vector embeddings
"""
import os
from typing import List, Optional
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import hashlib

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using various models"""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers")
        
        if self.embedding_model == "openai" and self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.model = None
            self.model_name = "text-embedding-ada-002"
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
            return self._mock_embedding(text)
        
        try:
            if self.embedding_model == "openai":
                # Use OpenAI API
                response = openai.Embedding.create(
                    model=self.model_name,
                    input=text
                )
                return response['data'][0]['embedding']
            
            elif self.embedding_model == "sentence-transformers" and self.model:
                # Use sentence-transformers
                embedding = self.model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            
            else:
                # Fall back to mock embeddings
                return self._mock_embedding(text)
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return self._mock_embedding(text)
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        
        try:
            if self.embedding_model == "openai":
                # Batch process with OpenAI
                response = openai.Embedding.create(
                    model=self.model_name,
                    input=texts
                )
                return [item['embedding'] for item in response['data']]
            
            elif self.embedding_model == "sentence-transformers" and self.model:
                # Batch process with sentence-transformers
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                return embeddings.tolist()
            
            else:
                # Fall back to mock embeddings
                return [self._mock_embedding(text) for text in texts]
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return [self._mock_embedding(text) for text in texts]
    
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


# Global embedding service instance
embedding_service = EmbeddingService()