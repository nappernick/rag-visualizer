#!/usr/bin/env python3
"""Test Supabase connection and entity storage"""
import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

def test_supabase_connection():
    """Test Supabase connection and basic operations"""
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_API_KEY")
    
    print(f"Testing Supabase connection...")
    print(f"URL: {supabase_url}")
    print(f"Key: {supabase_key[:20]}..." if supabase_key else "Key: Not found")
    
    if not supabase_url or not supabase_key:
        print("❌ Supabase credentials not found in environment")
        return False
    
    try:
        # Create client
        client: Client = create_client(supabase_url, supabase_key)
        print("✅ Supabase client created successfully")
        
        # Test 1: Check if entities table exists by querying it
        print("\nTesting entities table...")
        try:
            result = client.table("entities").select("*").limit(1).execute()
            print(f"✅ Entities table exists. Current count: {len(result.data) if result.data else 0}")
        except Exception as e:
            print(f"❌ Entities table error: {e}")
            print("   Please run the create_graph_tables.sql script in Supabase SQL Editor")
            return False
        
        # Test 2: Check relationships table
        print("\nTesting relationships table...")
        try:
            result = client.table("relationships").select("*").limit(1).execute()
            print(f"✅ Relationships table exists. Current count: {len(result.data) if result.data else 0}")
        except Exception as e:
            print(f"❌ Relationships table error: {e}")
            return False
        
        # Test 3: Check rag_documents table
        print("\nTesting rag_documents table...")
        try:
            result = client.table("rag_documents").select("*").limit(1).execute()
            print(f"✅ rag_documents table exists. Current count: {len(result.data) if result.data else 0}")
        except Exception as e:
            print(f"❌ rag_documents table error: {e}")
            return False
        
        # Test 4: Insert test entity
        print("\nTesting entity insertion...")
        test_entity = {
            "id": "test_entity_123",
            "name": "Test Entity",
            "entity_type": "test",
            "document_ids": ["doc_123"],
            "frequency": 1,
            "metadata": {"test": True}
        }
        
        try:
            result = client.table("entities").upsert(test_entity).execute()
            print("✅ Test entity inserted successfully")
            
            # Clean up test entity
            client.table("entities").delete().eq("id", "test_entity_123").execute()
            print("✅ Test entity cleaned up")
        except Exception as e:
            print(f"❌ Entity insertion error: {e}")
            return False
        
        print("\n✅ All Supabase tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        return False

if __name__ == "__main__":
    success = test_supabase_connection()
    sys.exit(0 if success else 1)