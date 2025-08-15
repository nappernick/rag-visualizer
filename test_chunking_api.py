#!/usr/bin/env python3
"""
Test script to evaluate chunking and graph extraction via API
"""
import requests
import json
import time
from typing import Dict, List, Any

API_BASE = "http://localhost:8642"

def test_document_upload():
    """Test document upload with sample text"""
    
    # Sample document with clear semantic structure
    sample_text = """
    # Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.
    
    ## Types of Machine Learning
    
    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Each type has its own characteristics and use cases.
    
    ### Supervised Learning
    
    In supervised learning, the algorithm learns from labeled training data. The system is provided with input-output pairs and learns to map inputs to outputs. Common algorithms include decision trees, random forests, and neural networks. Applications include image classification, spam detection, and medical diagnosis.
    
    ### Unsupervised Learning
    
    Unsupervised learning works with unlabeled data to discover hidden patterns. The algorithm tries to find structure in the data without predefined categories. Clustering algorithms like K-means and hierarchical clustering are common examples. This is useful for customer segmentation, anomaly detection, and data compression.
    
    ### Reinforcement Learning
    
    Reinforcement learning involves an agent learning to make decisions by interacting with an environment. The agent receives rewards or penalties for its actions and learns to maximize cumulative reward. This approach is used in game playing, robotics, and autonomous vehicles.
    
    ## Key Concepts
    
    Important concepts in machine learning include features, training data, validation, and overfitting. Features are the measurable properties of the phenomena being observed. Training data is used to teach the algorithm. Validation ensures the model generalizes well. Overfitting occurs when a model learns the training data too well and performs poorly on new data.
    
    ## Applications
    
    Machine learning has numerous real-world applications. In healthcare, it's used for disease prediction and drug discovery. In finance, it powers fraud detection and algorithmic trading. In technology, it enables recommendation systems, natural language processing, and computer vision. The field continues to evolve rapidly with new breakthroughs in deep learning and neural networks.
    """
    
    # Upload document
    response = requests.post(
        f"{API_BASE}/api/documents/upload",
        json={
            "title": "ML Overview Test",
            "content": sample_text,
            "metadata": {"test": True, "purpose": "chunking_evaluation"}
        }
    )
    
    if response.status_code == 200:
        doc_data = response.json()
        print(f"✅ Document uploaded successfully")
        print(f"   Document ID: {doc_data['id']}")
        print(f"   Title: {doc_data['title']}")
        return doc_data['id']
    else:
        print(f"❌ Failed to upload document: {response.status_code}")
        print(response.text)
        return None


def analyze_chunks(doc_id: str):
    """Analyze chunks for a document"""
    
    # Get chunks
    response = requests.get(f"{API_BASE}/api/documents/{doc_id}/chunks")
    
    if response.status_code != 200:
        print(f"❌ Failed to get chunks: {response.status_code}")
        return
    
    chunks = response.json()
    
    print(f"\n📊 Chunk Analysis for Document {doc_id}")
    print(f"   Total chunks: {len(chunks)}")
    
    if chunks:
        # Analyze chunk sizes
        sizes = [len(chunk.get('content', '')) for chunk in chunks]
        tokens = [chunk.get('tokens', 0) for chunk in chunks]
        
        print(f"\n   Character sizes:")
        print(f"     Min: {min(sizes)}")
        print(f"     Max: {max(sizes)}")
        print(f"     Avg: {sum(sizes) / len(sizes):.1f}")
        
        print(f"\n   Token counts:")
        print(f"     Min: {min(tokens)}")
        print(f"     Max: {max(tokens)}")
        print(f"     Avg: {sum(tokens) / len(tokens):.1f}")
        
        # Check for overlap
        print(f"\n   Chunk samples:")
        for i, chunk in enumerate(chunks[:3]):  # First 3 chunks
            content = chunk.get('content', '')
            print(f"\n   Chunk {i}:")
            print(f"     Type: {chunk.get('chunk_type', 'unknown')}")
            print(f"     Tokens: {chunk.get('tokens', 0)}")
            print(f"     Content preview: {content[:150]}...")
            if i > 0:
                # Check overlap with previous chunk
                prev_content = chunks[i-1].get('content', '')
                if prev_content and content:
                    # Simple overlap check - last part of prev in current
                    overlap_check = prev_content[-50:] in content
                    print(f"     Has overlap with previous: {overlap_check}")
    
    return chunks


def analyze_graph(doc_id: str):
    """Analyze graph entities and relationships"""
    
    # Get entities
    entities_response = requests.get(f"{API_BASE}/api/graph/entities")
    relationships_response = requests.get(f"{API_BASE}/api/graph/relationships")
    
    if entities_response.status_code != 200:
        print(f"❌ Failed to get entities: {entities_response.status_code}")
        return
    
    if relationships_response.status_code != 200:
        print(f"❌ Failed to get relationships: {relationships_response.status_code}")
        return
    
    all_entities = entities_response.json()
    all_relationships = relationships_response.json()
    
    # Filter for our document
    doc_entities = [e for e in all_entities if e.get('document_id') == doc_id]
    doc_relationships = [r for r in all_relationships if r.get('document_id') == doc_id]
    
    print(f"\n🔗 Graph Analysis for Document {doc_id}")
    print(f"   Total entities: {len(doc_entities)}")
    print(f"   Total relationships: {len(doc_relationships)}")
    
    if doc_entities:
        # Analyze entity types
        entity_types = {}
        for entity in doc_entities:
            etype = entity.get('entity_type', 'unknown')
            entity_types[etype] = entity_types.get(etype, 0) + 1
        
        print(f"\n   Entity types:")
        for etype, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
            print(f"     {etype}: {count}")
        
        # Sample entities
        print(f"\n   Sample entities:")
        for entity in doc_entities[:5]:
            print(f"     - {entity.get('name', 'unnamed')} ({entity.get('entity_type', 'unknown')})")
            if entity.get('description'):
                print(f"       Description: {entity['description'][:100]}...")
    
    if doc_relationships:
        # Analyze relationship types
        rel_types = {}
        for rel in doc_relationships:
            rtype = rel.get('relationship_type', 'unknown')
            rel_types[rtype] = rel_types.get(rtype, 0) + 1
        
        print(f"\n   Relationship types:")
        for rtype, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True):
            print(f"     {rtype}: {count}")
        
        # Sample relationships
        print(f"\n   Sample relationships:")
        for rel in doc_relationships[:5]:
            source = rel.get('source_entity', 'unknown')
            target = rel.get('target_entity', 'unknown')
            rel_type = rel.get('relationship_type', 'unknown')
            print(f"     - {source} --[{rel_type}]--> {target}")
    
    return doc_entities, doc_relationships


def test_search_capabilities(doc_id: str):
    """Test search and retrieval capabilities"""
    
    # Test vector search
    query = "What are the types of machine learning?"
    
    print(f"\n🔍 Testing Search Capabilities")
    print(f"   Query: '{query}'")
    
    # Vector search
    vector_response = requests.post(
        f"{API_BASE}/api/search/vector",
        json={"query": query, "top_k": 3}
    )
    
    if vector_response.status_code == 200:
        results = vector_response.json()
        print(f"\n   Vector search results: {len(results.get('results', []))} chunks found")
        for i, result in enumerate(results.get('results', [])[:3]):
            score = result.get('score', 0)
            content = result.get('content', '')
            print(f"     {i+1}. Score: {score:.3f}")
            print(f"        Preview: {content[:100]}...")
    
    # Graph search
    graph_response = requests.post(
        f"{API_BASE}/api/search/graph",
        json={"query": query, "max_depth": 2}
    )
    
    if graph_response.status_code == 200:
        results = graph_response.json()
        print(f"\n   Graph search results:")
        print(f"     Entities found: {len(results.get('entities', []))}")
        print(f"     Relationships found: {len(results.get('relationships', []))}")
        
        if results.get('entities'):
            print(f"     Sample entities:")
            for entity in results['entities'][:3]:
                print(f"       - {entity.get('name', 'unnamed')} ({entity.get('entity_type', 'unknown')})")


def main():
    """Run all tests"""
    print("=" * 60)
    print("RAG Visualizer - Chunking & Graph Analysis Test")
    print("=" * 60)
    
    # Test document upload
    doc_id = test_document_upload()
    
    if not doc_id:
        print("❌ Failed to upload document, exiting...")
        return
    
    # Wait for processing
    print("\n⏳ Waiting for document processing...")
    time.sleep(3)
    
    # Analyze chunks
    chunks = analyze_chunks(doc_id)
    
    # Analyze graph
    entities, relationships = analyze_graph(doc_id)
    
    # Test search
    test_search_capabilities(doc_id)
    
    print("\n" + "=" * 60)
    print("Test completed!")


if __name__ == "__main__":
    main()