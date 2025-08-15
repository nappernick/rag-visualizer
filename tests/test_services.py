#!/usr/bin/env python3
"""Test all external service connections"""
import os
import sys
import asyncio
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables from .env file (force reload)
load_dotenv(override=True)

async def test_services():
    print("Testing RAG Visualizer Service Connections\n")
    print("=" * 50)
    
    # Test OpenAI
    print("\n1. Testing OpenAI Connection...")
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
            # Test with a simple embedding
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input="test"
            )
            print(f"✅ OpenAI connected! Embedding dimension: {len(response.data[0].embedding)}")
        else:
            print("❌ OpenAI API key not found in .env")
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
    
    # Test Qdrant
    print("\n2. Testing Qdrant Connection...")
    try:
        from qdrant_client import QdrantClient
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if qdrant_url and qdrant_api_key:
            # Clean URL - remove port for Qdrant Cloud
            qdrant_url = qdrant_url.replace(":6333", "")
            
            client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=30,
                https=True,
                prefer_grpc=False
            )
            collections = client.get_collections()
            print(f"✅ Qdrant connected! Collections: {[c.name for c in collections.collections]}")
        else:
            print("❌ Qdrant credentials not found in .env")
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
    
    # Test Neo4j
    print("\n3. Testing Neo4j Connection...")
    try:
        from neo4j import GraphDatabase
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if neo4j_uri and neo4j_password:
            driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_password)
            )
            driver.verify_connectivity()
            with driver.session() as session:
                result = session.run("RETURN 1 AS test")
                test_val = result.single()["test"]
            driver.close()
            print(f"✅ Neo4j connected! Test query returned: {test_val}")
        else:
            print("❌ Neo4j credentials not found in .env")
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
    
    # Test Supabase
    print("\n4. Testing Supabase Connection...")
    try:
        from supabase import create_client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_API_KEY")
        
        if supabase_url and supabase_key:
            client = create_client(supabase_url, supabase_key)
            # Try to list tables or do a simple query
            result = client.table("documents").select("id").limit(1).execute()
            print(f"✅ Supabase connected!")
        else:
            print("⚠️  Supabase credentials not found in .env (optional service)")
    except Exception as e:
        if "does not exist" in str(e).lower():
            print("⚠️  Supabase connected but tables not created yet")
        else:
            print(f"⚠️  Supabase not configured: {e}")
    
    print("\n" + "=" * 50)
    print("\nService Configuration Summary:")
    print(f"- OpenAI API Key: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
    print(f"- Qdrant URL: {'✅ Set' if os.getenv('QDRANT_URL') else '❌ Missing'}")
    print(f"- Neo4j URI: {'✅ Set' if os.getenv('NEO4J_URI') else '❌ Missing'}")
    print(f"- Supabase URL: {'✅ Set' if os.getenv('SUPABASE_URL') else '⚠️  Not set (optional)'}")

if __name__ == "__main__":
    asyncio.run(test_services())