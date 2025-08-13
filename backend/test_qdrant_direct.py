#!/usr/bin/env python3
"""Test Qdrant connection with direct API"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_qdrant_api():
    """Test Qdrant using direct HTTP API"""
    
    # Get credentials
    qdrant_url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    # Keep the URL as is (with port if present)
    base_url = qdrant_url
        
    print(f"Testing Qdrant API at: {base_url}")
    
    # Test collections endpoint
    headers = {
        "api-key": api_key,  # Try both header formats
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        # List collections
        response = requests.get(f"{base_url}/collections", headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Collections: {data}")
            
            # Try to create our collection
            collection_name = "rag_visualizer_chunks"
            create_payload = {
                "vectors": {
                    "size": 1536,
                    "distance": "Cosine"
                }
            }
            
            create_response = requests.put(
                f"{base_url}/collections/{collection_name}",
                headers=headers,
                json=create_payload,
                timeout=10
            )
            
            if create_response.status_code in [200, 201]:
                print(f"✅ Created collection: {collection_name}")
            else:
                print(f"Collection response: {create_response.status_code} - {create_response.text}")
                
        else:
            print(f"❌ Failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_qdrant_api()