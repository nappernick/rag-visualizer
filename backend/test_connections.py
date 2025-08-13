#!/usr/bin/env python3
"""Test all service connections"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_supabase():
    """Test Supabase connection"""
    print("\n🔍 Testing Supabase connection...")
    try:
        from supabase import create_client, Client
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_API_KEY")
        
        if not url or not key:
            print("❌ Supabase credentials not found in .env")
            return False
            
        client = create_client(url, key)
        
        # Try to query tables
        response = client.table("documents").select("*").limit(1).execute()
        print(f"✅ Supabase connected successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
        return False

def test_qdrant():
    """Test Qdrant connection"""
    print("\n🔍 Testing Qdrant connection...")
    try:
        from qdrant_client import QdrantClient
        
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        
        if not url or not api_key:
            print("❌ Qdrant credentials not found in .env")
            return False
            
        client = QdrantClient(url=url, api_key=api_key)
        
        # Get collections
        collections = client.get_collections()
        print(f"✅ Qdrant connected successfully! Collections: {[c.name for c in collections.collections]}")
        return True
        
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return False

def test_neo4j():
    """Test Neo4j connection"""
    print("\n🔍 Testing Neo4j connection...")
    try:
        from neo4j import GraphDatabase
        
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not uri or not user or not password:
            print("❌ Neo4j credentials not found in .env")
            return False
            
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Test connectivity
        driver.verify_connectivity()
        driver.close()
        
        print(f"✅ Neo4j connected successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False

def test_openai():
    """Test OpenAI API connection"""
    print("\n🔍 Testing OpenAI API...")
    try:
        import openai
        
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            print("❌ OpenAI API key not found in .env")
            return False
            
        client = openai.OpenAI(api_key=api_key)
        
        # Test with a small embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="test"
        )
        
        print(f"✅ OpenAI API connected successfully! Embedding dimension: {len(response.data[0].embedding)}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {e}")
        return False

async def test_document_upload():
    """Test document upload flow"""
    print("\n🔍 Testing document upload flow...")
    try:
        import aiohttp
        import json
        
        # Create a test document
        test_content = "This is a test document for RAG visualizer. It contains information about testing the system."
        
        # Create form data
        data = aiohttp.FormData()
        data.add_field('file',
                      test_content.encode('utf-8'),
                      filename='test.txt',
                      content_type='text/plain')
        
        async with aiohttp.ClientSession() as session:
            async with session.post('http://localhost:8745/api/documents/upload', data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Document upload successful! Document ID: {result.get('id')}")
                    return True
                else:
                    text = await response.text()
                    print(f"❌ Document upload failed: {response.status} - {text}")
                    return False
                    
    except Exception as e:
        print(f"❌ Document upload test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("=" * 50)
    print("RAG Visualizer Integration Tests")
    print("=" * 50)
    
    results = {
        "Supabase": test_supabase(),
        "Qdrant": test_qdrant(),
        "Neo4j": test_neo4j(),
        "OpenAI": test_openai(),
    }
    
    # Test document upload if backend is running
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8745/health') as response:
                if response.status == 200:
                    results["Document Upload"] = await test_document_upload()
                else:
                    print("\n⚠️  Backend not responding on port 8745")
    except:
        print("\n⚠️  Backend not running - skipping upload test")
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    for service, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {service}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)