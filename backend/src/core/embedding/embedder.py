"""
Text embedding service for RAG pipeline
"""
from typing import List, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Handles text embedding generation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        embedding_config = config.get('embedding', {})
        
        self.provider = embedding_config.get('provider', 'openai')
        self.model = embedding_config.get('model', 'text-embedding-3-small')
        self.batch_size = embedding_config.get('batch_size', 100)
        self.embedding_dim = config.get('vector_store', {}).get('embedding_dim', 1536)
        
        # Initialize embedding provider
        self._init_provider()
    
    def _init_provider(self):
        """Initialize the embedding provider"""
        if self.provider == 'openai':
            try:
                import openai
                self.client = openai
                logger.info(f"Initialized OpenAI embeddings with model {self.model}")
            except ImportError:
                logger.warning("OpenAI not available, using mock embeddings")
                self.client = None
        elif self.provider == 'local':
            # Use sentence-transformers for local embeddings
            try:
                from sentence_transformers import SentenceTransformer
                self.client = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Initialized local sentence-transformer embeddings")
            except ImportError:
                logger.warning("Sentence-transformers not available, using mock embeddings")
                self.client = None
        else:
            logger.warning(f"Unknown provider {self.provider}, using mock embeddings")
            self.client = None
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        if self.client is None:
            # Return mock embeddings for testing
            return self._generate_mock_embeddings(texts)
        
        embeddings = []
        
        if self.provider == 'openai':
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error generating OpenAI embeddings: {e}")
                    # Fallback to mock embeddings
                    embeddings.extend(self._generate_mock_embeddings(batch))
        
        elif self.provider == 'local':
            try:
                # Sentence transformers can handle batches directly
                embeddings = self.client.encode(texts).tolist()
            except Exception as e:
                logger.error(f"Error generating local embeddings: {e}")
                embeddings = self._generate_mock_embeddings(texts)
        
        else:
            embeddings = self._generate_mock_embeddings(texts)
        
        return embeddings
    
    def _generate_mock_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate mock embeddings for testing
        
        Args:
            texts: Input texts
            
        Returns:
            Mock embedding vectors
        """
        embeddings = []
        
        for text in texts:
            # Generate deterministic mock embedding based on text
            np.random.seed(hash(text) % 2**32)
            embedding = np.random.randn(self.embedding_dim).tolist()
            
            # Normalize to unit length
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (np.array(embedding) / norm).tolist()
            
            embeddings.append(embedding)
        
        return embeddings
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure result is in [-1, 1] due to floating point errors
        return float(np.clip(similarity, -1.0, 1.0))
    
    def find_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[tuple[int, float]]:
        """
        Find most similar embeddings to a query
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []
        
        for idx, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            if similarity >= threshold:
                similarities.append((idx, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]