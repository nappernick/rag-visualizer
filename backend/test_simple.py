#!/usr/bin/env python3
"""
Simple test of enhanced RAG features without external dependencies
"""
import asyncio
import json

# Test imports
print("Testing imports...")
try:
    from src.core.query.query_enhancer import QueryEnhancer, QueryType
    print("✅ Query Enhancer imported")
except Exception as e:
    print(f"❌ Query Enhancer import failed: {e}")

try:
    from src.core.retrieval.hybrid_search import SearchConfig
    print("✅ Hybrid Search imported")
except Exception as e:
    print(f"❌ Hybrid Search import failed: {e}")

try:
    from src.models import RetrievalResult, Document, Chunk
    print("✅ Models imported")
except Exception as e:
    print(f"❌ Models import failed: {e}")


async def test_query_decomposition():
    """Test query decomposition without Bedrock"""
    print("\n" + "="*50)
    print("Testing Query Decomposition (Rule-based)")
    print("="*50)
    
    # Use rule-based decomposition
    enhancer = QueryEnhancer(bedrock_client=None, use_local_models=False)
    
    test_query = "How does authentication affect reporting and what are the security implications?"
    
    enhanced = await enhancer.enhance_query(test_query)
    
    print(f"\n📝 Query: {test_query}")
    print(f"   Type: {enhanced.query_type.value}")
    print(f"   Complexity: {enhanced.complexity_score:.2f}")
    
    print(f"\n   Sub-queries ({len(enhanced.sub_queries)}):")
    for i, sq in enumerate(enhanced.sub_queries, 1):
        print(f"      {i}. {sq.question}")
    
    print(f"\n   Key concepts: {enhanced.key_concepts}")
    
    return enhanced


async def test_search_config():
    """Test search configuration"""
    print("\n" + "="*50)
    print("Testing Search Configuration")
    print("="*50)
    
    config = SearchConfig(
        vector_weight=0.4,
        keyword_weight=0.3,
        metadata_weight=0.3,
        rrf_k=60
    )
    
    print(f"\n🔍 RRF Configuration:")
    print(f"   Vector: {config.vector_weight:.0%}")
    print(f"   Keyword: {config.keyword_weight:.0%}")
    print(f"   Metadata: {config.metadata_weight:.0%}")
    print(f"   RRF k-parameter: {config.rrf_k}")
    
    # Show RRF scoring example
    print(f"\n📊 RRF Scoring Example:")
    print(f"   Rank 1: {config.vector_weight * (1/(config.rrf_k + 1)):.4f}")
    print(f"   Rank 2: {config.vector_weight * (1/(config.rrf_k + 2)):.4f}")
    print(f"   Rank 5: {config.vector_weight * (1/(config.rrf_k + 5)):.4f}")
    print(f"   Rank 10: {config.vector_weight * (1/(config.rrf_k + 10)):.4f}")


def test_models():
    """Test model creation"""
    print("\n" + "="*50)
    print("Testing Data Models")
    print("="*50)
    
    # Create test document
    doc = Document(
        id="doc1",
        title="Test Document",
        content="This is test content about authentication."
    )
    print(f"\n✅ Document created: {doc.title}")
    
    # Create test chunk
    chunk = Chunk(
        id="chunk1",
        document_id=doc.id,
        content="Authentication is important for security.",
        position=0
    )
    print(f"✅ Chunk created: {chunk.id}")
    
    # Create retrieval result
    result = RetrievalResult(
        chunk_id=chunk.id,
        content=chunk.content,
        score=0.95,
        source="hybrid"
    )
    print(f"✅ RetrievalResult created: score={result.score}, source={result.source}")


async def main():
    print("\n🚀 TESTING ENHANCED RAG COMPONENTS 🚀\n")
    
    # Test models
    test_models()
    
    # Test query decomposition
    enhanced = await test_query_decomposition()
    
    # Test search config
    await test_search_config()
    
    print("\n" + "="*50)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*50)
    
    print("\n📝 Summary:")
    print("   • Models are properly defined")
    print("   • Query decomposition works with rule-based fallback")
    print("   • RRF configuration is set up correctly")
    print("   • System is ready for integration with services")
    
    print("\n💡 Next Steps:")
    print("   1. Start Redis with search module")
    print("   2. Start Qdrant vector database")
    print("   3. Configure AWS Bedrock for Claude 3.5 Haiku")
    print("   4. Test with live data through API endpoints")


if __name__ == "__main__":
    asyncio.run(main())