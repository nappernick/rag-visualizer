#!/usr/bin/env python3
"""
Test script to verify entity and relationship extraction
"""
import requests
import json
import sys

API_BASE = "http://localhost:8745/api"

def test_graph_extraction():
    """Test entity and relationship extraction"""
    
    # Sample document data
    test_document = {
        "document_id": "test_doc_123",
        "chunks": [
            {
                "id": "chunk_1",
                "content": """
                Amazon Web Services (AWS) provides cloud computing services including EC2 for compute, 
                S3 for storage, and RDS for databases. AWS Lambda enables serverless computing.
                Companies like Netflix and Airbnb use AWS for their infrastructure.
                AWS integrates with Docker for containerization and Kubernetes for orchestration.
                """
            }
        ],
        "extract_entities": True,
        "extract_relationships": True,
        "use_claude": True
    }
    
    print("Testing entity and relationship extraction...")
    print(f"Sending request to {API_BASE}/graph/extract")
    
    try:
        response = requests.post(
            f"{API_BASE}/graph/extract",
            json=test_document,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Success! Status: {response.status_code}")
            print(f"\nExtracted {len(data['entities'])} entities:")
            for entity in data['entities']:
                print(f"  - {entity['name']} ({entity['entity_type']})")
            
            print(f"\nExtracted {len(data['relationships'])} relationships:")
            for rel in data['relationships']:
                # Find entity names
                source_name = next((e['name'] for e in data['entities'] if e['id'] == rel['source_entity_id']), rel['source_entity_id'])
                target_name = next((e['name'] for e in data['entities'] if e['id'] == rel['target_entity_id']), rel['target_entity_id'])
                print(f"  - {source_name} --[{rel['relationship_type']}]--> {target_name}")
            
            # Check for the problematic all-pairs issue
            entity_count = len(data['entities'])
            relationship_count = len(data['relationships'])
            max_possible = (entity_count * (entity_count - 1)) / 2
            
            print(f"\n📊 Statistics:")
            print(f"  - Entities: {entity_count}")
            print(f"  - Relationships: {relationship_count}")
            print(f"  - Max possible relationships (all pairs): {max_possible:.0f}")
            
            if relationship_count == max_possible and entity_count > 5:
                print("  ⚠️  WARNING: Relationship count equals all possible pairs!")
                print("     This suggests automatic relationship creation between all entities.")
            else:
                print("  ✅ Relationship count looks reasonable")
                
        else:
            print(f"\n❌ Error! Status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Request failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_graph_extraction()
    sys.exit(0 if success else 1)