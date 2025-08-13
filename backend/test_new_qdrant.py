#!/usr/bin/env python3
"""Test new Qdrant US cluster"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv('.env', override=True)

def test_qdrant():
    """Test Qdrant connection"""
    try:
        # Get credentials
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        print(f"Connecting to Qdrant at: {qdrant_url}")
        
        # Initialize client exactly as shown in the example
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        # List collections
        print("Fetching collections...")
        collections = client.get_collections()
        print(f"✅ Connected! Existing collections: {[c.name for c in collections.collections]}")
        
        # Create our collection
        collection_name = "rag_visualizer_chunks"
        
        # Check if collection exists
        collection_exists = collection_name in [c.name for c in collections.collections]
        
        if not collection_exists:
            print(f"Creating collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,  # text-embedding-3-small dimension
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Created collection: {collection_name}")
        else:
            print(f"✅ Collection already exists: {collection_name}")
        
        # Verify collection
        collection_info = client.get_collection(collection_name)
        print(f"Collection info: Points count = {collection_info.points_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_qdrant()