"""
Configuration module for RAG Visualizer backend.
Handles environment variables and service configuration.
"""
import os
from typing import Optional
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in project root
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv is already installed, but just in case


class ServiceConfig:
    """Service availability and connection configuration"""
    
    # Service toggles
    ENABLE_REDIS: bool = os.getenv("ENABLE_REDIS", "false").lower() == "true"
    ENABLE_QDRANT_LOCAL: bool = os.getenv("ENABLE_QDRANT_LOCAL", "false").lower() == "true"
    ENABLE_NEO4J: bool = os.getenv("ENABLE_NEO4J", "false").lower() == "true"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./rag_visualizer.db")
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    
    # Qdrant
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL") or None
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY") or None
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    
    # Neo4j
    NEO4J_URL: str = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    
    # Supabase
    SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL") or None
    SUPABASE_ANON_API_KEY: Optional[str] = os.getenv("SUPABASE_ANON_API_KEY") or None
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY") or None
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "text-embedding-ada-002")
    
    # Development
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def is_redis_enabled(cls) -> bool:
        """Check if Redis should be used"""
        return cls.ENABLE_REDIS
    
    @classmethod 
    def is_qdrant_enabled(cls) -> bool:
        """Check if Qdrant should be used"""
        return cls.ENABLE_QDRANT_LOCAL or (cls.QDRANT_URL and cls.QDRANT_API_KEY)
    
    @classmethod
    def is_neo4j_enabled(cls) -> bool:
        """Check if Neo4j should be used"""
        return cls.ENABLE_NEO4J
    
    @classmethod
    def get_qdrant_config(cls) -> dict:
        """Get Qdrant configuration based on environment"""
        if cls.QDRANT_URL and cls.QDRANT_API_KEY:
            return {
                "url": cls.QDRANT_URL,
                "api_key": cls.QDRANT_API_KEY
            }
        elif cls.ENABLE_QDRANT_LOCAL:
            return {
                "host": cls.QDRANT_HOST,
                "port": cls.QDRANT_PORT
            }
        else:
            return {}


# Global configuration instance
config = ServiceConfig()