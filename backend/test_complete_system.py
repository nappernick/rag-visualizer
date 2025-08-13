#!/usr/bin/env python3
"""Complete system test for RAG Visualizer backend"""
import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Force reload environment
from dotenv import load_dotenv
load_dotenv(override=True)

async def test_system():
    print("=" * 70)
    print("RAG VISUALIZER COMPLETE SYSTEM TEST")
    print("=" * 70)
    
    # Import service manager
    from core.service_manager import ServiceManager
    
    # Verify environment variables
    print("\n1. ENVIRONMENT VARIABLES:")
    print("-" * 50)
    env_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "")[:20] + "..." if os.getenv("OPENAI_API_KEY") else "NOT SET",
        "QDRANT_URL": os.getenv("QDRANT_URL", "NOT SET"),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY", "")[:20] + "..." if os.getenv("QDRANT_API_KEY") else "NOT SET",
        "NEO4J_URI": os.getenv("NEO4J_URI", "NOT SET"),
        "NEO4J_PASSWORD": "***SET***" if os.getenv("NEO4J_PASSWORD") else "NOT SET",
        "SUPABASE_URL": os.getenv("SUPABASE_URL", "NOT SET"),
    }
    
    for key, value in env_vars.items():
        status = "✅" if value != "NOT SET" else "❌"
        print(f"{status} {key}: {value}")
    
    # Test all services
    print("\n2. SERVICE INITIALIZATION:")
    print("-" * 50)
    
    results = ServiceManager.verify_all_services()
    for service, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {service.capitalize()} Service: {'Connected' if status else 'Not Connected'}")
    
    # Test actual functionality
    print("\n3. FUNCTIONAL TESTS:")
    print("-" * 50)
    
    # Test document upload and processing pipeline
    from services.storage import get_storage_service
    from services.entity_service import get_entity_service
    from services.vector_service import get_vector_service
    from services.embedding_service import get_embedding_service
    from services.graph_service import get_graph_service
    
    test_doc = {
        "id": "test-doc-001",
        "title": "Test Document",
        "content": "OpenAI developed GPT-4, a large language model. It uses Qdrant for vector storage and Neo4j for graph databases.",
        "doc_type": "test"
    }
    
    # Test storage
    print("\n📦 Testing Storage Service...")
    storage = get_storage_service()
    stored_doc = await storage.store_document(test_doc)
    print(f"   Document stored: {stored_doc['id']}")
    
    # Test entity extraction
    print("\n🔍 Testing Entity Extraction...")
    entity_service = get_entity_service()
    entities = await entity_service.extract_entities(
        text=test_doc["content"],
        document_id=test_doc["id"],
        chunk_ids=["chunk-001"],
        use_claude=False
    )
    print(f"   Extracted {len(entities)} entities:")
    for e in entities[:3]:
        print(f"     - {e['name']} ({e['entity_type']})")
    
    # Test embeddings
    print("\n🧮 Testing Embedding Service...")
    embedding_service = get_embedding_service()
    try:
        embedding = await embedding_service.generate_embedding("test text")
        print(f"   Generated embedding with dimension: {len(embedding)}")
    except Exception as e:
        print(f"   ❌ Embedding failed: {e}")
    
    # Test vector storage
    print("\n📊 Testing Vector Storage...")
    vector_service = get_vector_service()
    if vector_service.initialized:
        chunks = [{"id": "chunk-001", "content": test_doc["content"], "document_id": test_doc["id"]}]
        try:
            embeddings = await embedding_service.generate_embeddings([c["content"] for c in chunks])
            success = await vector_service.store_vectors(chunks, embeddings)
            print(f"   Vectors stored: {success}")
        except Exception as e:
            print(f"   Vector storage issue: {e}")
    else:
        print("   ⚠️ Vector service not initialized")
    
    # Test graph storage
    print("\n🕸️ Testing Graph Storage...")
    graph_service = get_graph_service()
    if graph_service.initialized and entities:
        success = await graph_service.store_entities(entities)
        print(f"   Entities stored in graph: {success}")
        
        # Retrieve them
        retrieved = await graph_service.get_document_entities(test_doc["id"])
        print(f"   Retrieved {len(retrieved)} entities from graph")
    else:
        print("   ⚠️ Graph service not initialized or no entities")
    
    # Test search functionality
    print("\n🔎 Testing Search...")
    if vector_service.initialized:
        try:
            query_embedding = await embedding_service.generate_embedding("GPT-4 language model")
            results = await vector_service.search_similar(query_embedding, limit=5)
            print(f"   Found {len(results)} similar chunks")
        except Exception as e:
            print(f"   Search issue: {e}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_system())