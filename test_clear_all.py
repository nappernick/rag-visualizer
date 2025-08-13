#!/usr/bin/env python3
"""Test the Clear All Data functionality"""
import requests
import sys
from check_stored_data import check_stored_data

API_BASE = "http://localhost:8745/api"

def test_clear_all():
    """Test clearing all data from all services"""
    
    print("🧹 Testing Clear All Data functionality...")
    print("=" * 50)
    
    # Show current data
    print("\n📊 Current data before clearing:")
    check_stored_data()
    
    # Call clear all endpoint
    print("\n" + "=" * 50)
    print("🚨 Clearing all data...")
    
    try:
        response = requests.delete(f"{API_BASE}/clear-all")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Status: {response.status_code}")
            print("\n📋 Cleared data summary:")
            
            if "cleared" in data:
                cleared = data["cleared"]
                print(f"  📄 Documents: {cleared.get('documents', 0)}")
                print(f"  📦 Chunks: {cleared.get('chunks', 0)}")
                print(f"  🔵 Supabase Entities: {cleared.get('supabase_entities', 0)}")
                print(f"  🔗 Supabase Relationships: {cleared.get('supabase_relationships', 0)}")
                print(f"  🌐 Neo4j Nodes: {cleared.get('neo4j_nodes', 0)}")
                print(f"  🔗 Neo4j Relationships: {cleared.get('neo4j_relationships', 0)}")
                print(f"  📍 Vectors: {cleared.get('vectors', 'N/A')}")
            
            print(f"\n💬 Message: {data.get('message', 'No message')}")
            
            # Verify data was cleared
            print("\n" + "=" * 50)
            print("📊 Verifying data after clearing:")
            check_stored_data()
            
            return True
        else:
            print(f"❌ Error! Status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    success = test_clear_all()
    sys.exit(0 if success else 1)