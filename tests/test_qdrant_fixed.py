#!/usr/bin/env python3
"""Test Qdrant connection with fixed URL"""

from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

print(f"Testing Qdrant connection to: {url}")

try:
    # Create client without port in URL
    client = QdrantClient(
        url=url,
        api_key=api_key,
        timeout=30
    )
    
    # Test connection
    collections = client.get_collections()
    print(f"✅ Connection successful!")
    print(f"Collections found: {[c.name for c in collections.collections]}")
    
    # Get collection info if it exists
    collection_name = os.getenv("QDRANT_COLLECTION", "chunks")
    for c in collections.collections:
        if c.name == collection_name:
            info = client.get_collection(collection_name)
            print(f"\nCollection '{collection_name}' info:")
            print(f"  - Points count: {info.points_count}")
            print(f"  - Vector size: {info.config.params.vectors.size}")
            print(f"  - Status: {info.status}")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")