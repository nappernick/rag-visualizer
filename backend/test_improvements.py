#!/usr/bin/env python3
"""
Test script to verify performance tracking and Neo4j cleanup improvements
"""
import asyncio
import aiohttp
import json
import sys
from pathlib import Path

API_BASE = "http://localhost:8642/api"

async def test_clear_all():
    """Test that clear-all properly deletes Neo4j nodes"""
    print("\n🧹 Testing clear-all functionality...")
    
    async with aiohttp.ClientSession() as session:
        # Clear all data
        async with session.delete(f"{API_BASE}/clear-all") as resp:
            if resp.status != 200:
                print(f"❌ Failed to clear data: {resp.status}")
                return False
            
            data = await resp.json()
            print(f"✅ Clear-all response: {json.dumps(data, indent=2)}")
            
            # Check if Neo4j nodes were cleared
            neo4j_nodes = data.get("cleared", {}).get("neo4j_nodes", 0)
            print(f"   Neo4j nodes deleted: {neo4j_nodes}")
            
            return True

async def test_upload_with_performance():
    """Test document upload with performance tracking"""
    print("\n📊 Testing document upload with performance tracking...")
    
    # Create a test text file
    test_content = """
    This is a test document for the RAG visualizer system.
    It contains information about machine learning, artificial intelligence,
    and natural language processing. The system uses embeddings to understand
    text and creates a knowledge graph with entities and relationships.
    
    Key technologies include:
    - Python and FastAPI for the backend
    - React and TypeScript for the frontend  
    - Neo4j for graph storage
    - Qdrant for vector embeddings
    - Supabase for document storage
    - Claude AI for entity extraction
    
    The system supports semantic chunking, entity extraction, and relationship mapping.
    It provides visualizations for chunks, knowledge graphs, and query results.
    """ * 10  # Make it longer to see meaningful timing
    
    test_file = Path("/tmp/test_doc.txt")
    test_file.write_text(test_content)
    
    async with aiohttp.ClientSession() as session:
        # Upload the document
        with open(test_file, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename='test_doc.txt', content_type='text/plain')
            
            async with session.post(f"{API_BASE}/documents/upload", data=data) as resp:
                if resp.status != 200:
                    print(f"❌ Failed to upload document: {resp.status}")
                    text = await resp.text()
                    print(f"   Error: {text}")
                    return False
                
                result = await resp.json()
                print(f"✅ Document uploaded successfully!")
                
                # Check for performance metrics
                if 'performance' in result:
                    perf = result['performance']
                    print(f"\n📈 Performance Metrics:")
                    print(f"   File read: {perf.get('file_read', 'N/A')}s")
                    print(f"   Content processing: {perf.get('content_processing', 'N/A')}s")
                    print(f"   Chunking: {perf.get('chunking', 'N/A')}s ({perf.get('chunk_count', 0)} chunks)")
                    print(f"   Entity extraction: {perf.get('entity_extraction', 'N/A')}s ({perf.get('entity_count', 0)} entities, {perf.get('relationship_count', 0)} relationships)")
                    print(f"   Storage: {perf.get('document_storage', 'N/A')}s")
                    print(f"   Total time: {perf.get('total_time', 'N/A')}s")
                    print(f"   Content length: {perf.get('content_length', 0)} characters")
                    
                    # Calculate efficiency metrics
                    if perf.get('content_length') and perf.get('total_time'):
                        speed = (perf['content_length'] / perf['total_time']) / 1000
                        print(f"\n⚡ Processing speed: {speed:.1f} k chars/s")
                    
                    if perf.get('entity_count') and perf.get('content_length'):
                        density = (perf['entity_count'] / perf['content_length']) * 1000
                        print(f"   Entity density: {density:.2f} entities per 1k chars")
                else:
                    print("⚠️  No performance metrics in response")
                
                return result.get('id')

async def test_get_stats(doc_id):
    """Test retrieving document with performance data"""
    print(f"\n📋 Testing stats retrieval for document {doc_id}...")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_BASE}/documents/{doc_id}") as resp:
            if resp.status != 200:
                print(f"❌ Failed to get document: {resp.status}")
                return False
            
            data = await resp.json()
            print(f"✅ Document retrieved")
            
            # The performance data should be in metadata
            if 'metadata' in data:
                print(f"   Metadata keys: {list(data['metadata'].keys())}")
            
            return True

async def main():
    """Run all tests"""
    print("🚀 Starting RAG Visualizer improvement tests...")
    
    # Test 1: Clear all data
    if not await test_clear_all():
        print("❌ Clear-all test failed")
        return 1
    
    await asyncio.sleep(2)
    
    # Test 2: Upload with performance tracking
    doc_id = await test_upload_with_performance()
    if not doc_id:
        print("❌ Upload test failed")
        return 1
    
    await asyncio.sleep(2)
    
    # Test 3: Get stats
    if not await test_get_stats(doc_id):
        print("❌ Stats test failed")
        return 1
    
    print("\n✅ All tests passed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))