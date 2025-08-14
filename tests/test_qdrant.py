#!/usr/bin/env python3
"""Test Qdrant connection and create collection"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qdrant():
    """Test Qdrant connection"""
    try:
        # Get credentials
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        logger.info(f"Connecting to Qdrant at: {qdrant_url}")
        
        # Remove port from URL if present (Qdrant Cloud uses default HTTPS port)
        if ":6333" in qdrant_url:
            qdrant_url = qdrant_url.replace(":6333", "")
        
        logger.info(f"Using URL: {qdrant_url}")
        
        # Initialize client with longer timeout
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60,  # 60 second timeout
            https=True,
            prefer_grpc=False  # Use HTTP instead of gRPC
        )
        
        # List collections
        logger.info("Fetching collections...")
        collections = client.get_collections()
        logger.info(f"Existing collections: {[c.name for c in collections.collections]}")
        
        # Create our collection
        collection_name = "rag_visualizer_chunks"
        
        # Check if collection exists
        collection_exists = collection_name in [c.name for c in collections.collections]
        
        if not collection_exists:
            logger.info(f"Creating collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,  # text-embedding-3-small dimension
                    distance=Distance.COSINE
                )
            )
            logger.info(f"✅ Created collection: {collection_name}")
        else:
            logger.info(f"✅ Collection already exists: {collection_name}")
        
        # Verify collection
        collection_info = client.get_collection(collection_name)
        logger.info(f"Collection info: {collection_info}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error connecting to Qdrant: {e}")
        return False

if __name__ == "__main__":
    if test_qdrant():
        logger.info("✅ Qdrant connection successful!")
    else:
        logger.error("❌ Qdrant connection failed!")