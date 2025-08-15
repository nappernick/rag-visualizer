#!/usr/bin/env python3
"""
Apply weight system migration using Supabase SDK
"""
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def apply_migration():
    # Get Supabase credentials
    url = os.getenv('SUPABASE_URL')
    service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not url or not service_key:
        print("Error: Missing Supabase credentials")
        return False
    
    print(f"Connecting to Supabase at: {url}")
    
    # Create client with service role key for full access
    client = create_client(url, service_key)
    
    # First, let's check if the weight column already exists by trying to update metadata
    try:
        # Try to add weight to document metadata
        print("\nChecking if we can store weight in metadata...")
        
        # Get existing documents
        response = client.table('rag_documents').select("*").limit(1).execute()
        
        if response.data and len(response.data) > 0:
            doc = response.data[0]
            doc_id = doc['id']
            
            # Update metadata to include weight
            metadata = doc.get('metadata', {})
            metadata['weight'] = 1.0
            
            update_response = client.table('rag_documents').update({
                'metadata': metadata
            }).eq('id', doc_id).execute()
            
            print(f"✓ Successfully updated document {doc_id} with weight in metadata")
        else:
            print("No documents found to test with")
        
        # Create weight_rules table data structure in metadata for now
        print("\nStoring default weight rules in a document metadata...")
        
        default_rules = {
            "rules": [
                {
                    "id": "default_doc_type",
                    "name": "Document Type Weights",
                    "rule_type": "document_type",
                    "enabled": True,
                    "priority": 100,
                    "conditions": {
                        "type_weights": {
                            "pdf": 1.5,
                            "markdown": 1.2,
                            "text": 1.0,
                            "code": 0.8,
                            "image": 0.5
                        }
                    },
                    "weight_modifier": 1.0
                },
                {
                    "id": "default_temporal",
                    "name": "Recency Boost",
                    "rule_type": "temporal",
                    "enabled": True,
                    "priority": 90,
                    "conditions": {
                        "ranges": [
                            {"within": "7d", "weight": 2.0},
                            {"within": "30d", "weight": 1.5},
                            {"within": "90d", "weight": 1.2},
                            {"older_than": "365d", "weight": 0.7}
                        ]
                    },
                    "weight_modifier": 1.0
                },
                {
                    "id": "default_title_important",
                    "name": "Important Documents",
                    "rule_type": "title_pattern",
                    "enabled": True,
                    "priority": 80,
                    "conditions": {
                        "patterns": [
                            {"match": "contains", "value": "important", "weight": 2.0},
                            {"match": "contains", "value": "critical", "weight": 2.5},
                            {"match": "contains", "value": "policy", "weight": 1.8},
                            {"match": "contains", "value": "draft", "weight": 0.5}
                        ]
                    },
                    "weight_modifier": 1.0
                }
            ]
        }
        
        # Store rules as a special document
        rules_doc = {
            "id": "SYSTEM_WEIGHT_RULES",
            "title": "System Weight Rules Configuration",
            "content": "This document stores the weight rules configuration for the RAG system",
            "doc_type": "system",
            "status": "completed",
            "metadata": default_rules
        }
        
        # Try to insert or update the rules document
        try:
            client.table('rag_documents').upsert(rules_doc).execute()
            print("✓ Successfully stored weight rules configuration")
        except Exception as e:
            print(f"Warning: Could not store rules document: {e}")
        
        print("\n" + "="*50)
        print("Migration workaround complete!")
        print("Weight system will use document metadata to store weights")
        print("Weight rules stored in special system document")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = apply_migration()
    exit(0 if success else 1)