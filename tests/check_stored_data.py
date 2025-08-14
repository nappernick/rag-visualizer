#!/usr/bin/env python3
"""Check what data is stored in Supabase"""
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import json

load_dotenv()

def check_stored_data():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_API_KEY")
    
    client: Client = create_client(supabase_url, supabase_key)
    
    # Check documents
    print("📄 Documents in Supabase:")
    docs = client.table("rag_documents").select("*").execute()
    for doc in docs.data:
        print(f"  - {doc['id']}: {doc['title']}")
    
    # Check entities
    print(f"\n🔵 Entities in Supabase ({len(client.table('entities').select('*').execute().data)} total):")
    entities = client.table("entities").select("*").limit(10).execute()
    for entity in entities.data:
        print(f"  - {entity['name']} ({entity['entity_type']})")
    
    # Check relationships
    print(f"\n🔗 Relationships in Supabase ({len(client.table('relationships').select('*').execute().data)} total):")
    relationships = client.table("relationships").select("*").limit(10).execute()
    for rel in relationships.data:
        print(f"  - {rel['source_entity_id']} --[{rel['relationship_type']}]--> {rel['target_entity_id']}")

if __name__ == "__main__":
    check_stored_data()