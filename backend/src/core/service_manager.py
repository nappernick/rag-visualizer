"""
Centralized Service Manager for RAG Visualizer
Ensures all services use latest environment variables and handles connection issues gracefully
"""
import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ServiceManager:
    """Manages all service instances with proper environment variable reloading"""
    
    _instances: Dict[str, Any] = {}
    _env_hash: Optional[str] = None
    
    @classmethod
    def reload_env(cls) -> bool:
        """Force reload environment variables from .env file"""
        # Find .env file
        env_path = Path(__file__).parent.parent.parent.parent / ".env"
        if not env_path.exists():
            logger.warning(f".env file not found at {env_path}")
            return False
        
        # Calculate hash of .env file to detect changes
        with open(env_path, 'rb') as f:
            import hashlib
            env_hash = hashlib.md5(f.read()).hexdigest()
        
        # Only reload if changed
        if env_hash != cls._env_hash:
            load_dotenv(env_path, override=True)
            cls._env_hash = env_hash
            cls._instances.clear()  # Clear all cached instances
            logger.info("✅ Reloaded environment variables from .env")
            return True
        return False
    
    @classmethod
    def get_service(cls, service_name: str, force_new: bool = False):
        """Get or create a service instance"""
        # Always check for env changes
        cls.reload_env()
        
        if force_new or service_name not in cls._instances:
            if service_name == "vector":
                cls._instances[service_name] = cls._create_vector_service()
            elif service_name == "graph":
                cls._instances[service_name] = cls._create_graph_service()
            elif service_name == "entity":
                cls._instances[service_name] = cls._create_entity_service()
            elif service_name == "embedding":
                cls._instances[service_name] = cls._create_embedding_service()
            elif service_name == "storage":
                cls._instances[service_name] = cls._create_storage_service()
            else:
                raise ValueError(f"Unknown service: {service_name}")
        
        return cls._instances[service_name]
    
    @classmethod
    def _create_vector_service(cls):
        """Create vector service with proper Qdrant configuration"""
        from ..services.vector_service import VectorService
        
        # Log current configuration
        url = os.getenv("QDRANT_URL", "")
        api_key = os.getenv("QDRANT_API_KEY", "")
        logger.info(f"Creating VectorService with Qdrant URL: {url[:50]}...")
        
        service = VectorService()
        
        # Verify connection
        if service.initialized:
            logger.info("✅ VectorService initialized with Qdrant")
        else:
            logger.warning("⚠️ VectorService running without Qdrant")
        
        return service
    
    @classmethod
    def _create_graph_service(cls):
        """Create graph service with proper Neo4j configuration"""
        from ..services.graph_service import GraphService
        
        # Log current configuration
        uri = os.getenv("NEO4J_URI", "")
        logger.info(f"Creating GraphService with Neo4j URI: {uri}")
        
        service = GraphService()
        
        # Verify connection
        if service.initialized:
            logger.info("✅ GraphService initialized with Neo4j")
        else:
            logger.warning("⚠️ GraphService running without Neo4j")
        
        return service
    
    @classmethod
    def _create_entity_service(cls):
        """Create entity extraction service"""
        from ..services.entity_service import EntityService
        logger.info("Creating EntityService")
        return EntityService()
    
    @classmethod
    def _create_embedding_service(cls):
        """Create embedding service with proper OpenAI configuration"""
        from ..services.embedding_service import EmbeddingService
        
        # Log current configuration
        api_key = os.getenv("OPENAI_API_KEY", "")
        model = os.getenv("EMBEDDING_MODEL", "sentence-transformers")
        logger.info(f"Creating EmbeddingService with model: {model}")
        
        service = EmbeddingService()
        
        # Verify configuration
        if service.embedding_model == "openai" and service.openai_client:
            logger.info("✅ EmbeddingService initialized with OpenAI")
        elif service.embedding_model == "sentence-transformers" and service.model:
            logger.info("✅ EmbeddingService initialized with Sentence Transformers")
        else:
            logger.warning("⚠️ EmbeddingService may not be properly configured")
        
        return service
    
    @classmethod
    def _create_storage_service(cls):
        """Create storage service with Supabase configuration"""
        from ..services.storage import StorageService
        
        # Log current configuration
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_ANON_API_KEY", "")
        
        if url and key:
            logger.info(f"Creating StorageService with Supabase URL: {url[:50]}...")
        else:
            logger.info("Creating StorageService without Supabase")
        
        service = StorageService()
        
        # Verify connection
        if service.initialized:
            logger.info("✅ StorageService initialized with Supabase")
        else:
            logger.warning("⚠️ StorageService running without persistence")
        
        return service
    
    @classmethod
    def verify_all_services(cls) -> Dict[str, bool]:
        """Verify all services are properly configured"""
        cls.reload_env()
        
        results = {}
        
        # Test each service
        for service_name in ["vector", "graph", "entity", "embedding", "storage"]:
            try:
                service = cls.get_service(service_name)
                if hasattr(service, 'initialized'):
                    results[service_name] = service.initialized
                else:
                    results[service_name] = True
            except Exception as e:
                logger.error(f"Failed to initialize {service_name}: {e}")
                results[service_name] = False
        
        return results


# Global function replacements for backward compatibility
def get_vector_service():
    """Get vector service instance"""
    return ServiceManager.get_service("vector")


def get_graph_service():
    """Get graph service instance"""
    return ServiceManager.get_service("graph")


def get_entity_service():
    """Get entity service instance"""
    return ServiceManager.get_service("entity")


def get_embedding_service():
    """Get embedding service instance"""
    return ServiceManager.get_service("embedding")


def get_storage_service():
    """Get storage service instance"""
    return ServiceManager.get_service("storage")