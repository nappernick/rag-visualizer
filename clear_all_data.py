#!/usr/bin/env python3
"""
Script to clear all data from cloud storage services
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add backend src to path
backend_src = Path(__file__).parent / "backend" / "src"
sys.path.insert(0, str(backend_src))

# Load environment variables
env_path = Path(__file__).parent / "backend" / ".env"
load_dotenv(env_path, override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clear_all_data():
    """Clear all data from all storage services"""
    
    logger.info("=" * 60)
    logger.info("STARTING DATA CLEARANCE PROCESS")
    logger.info("=" * 60)
    
    # Import after path setup
    from supabase import create_client
    from qdrant_client import QdrantClient
    from neo4j import GraphDatabase
    
    # 1. Clear Supabase
    logger.info("\n1. CLEARING SUPABASE...")
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_API_KEY")
        supabase = create_client(supabase_url, supabase_key)
        
        # Delete all chunks first (due to foreign key)
        logger.info("   Deleting all chunks from rag_chunks table...")
        chunks_result = supabase.table("rag_chunks").delete().neq("id", "").execute()
        logger.info(f"   ✓ Deleted chunks from Supabase")
        
        # Delete all documents
        logger.info("   Deleting all documents from rag_documents table...")
        docs_result = supabase.table("rag_documents").delete().neq("id", "").execute()
        logger.info(f"   ✓ Deleted documents from Supabase")
        
        # Verify deletion
        remaining_chunks = supabase.table("rag_chunks").select("count", count="exact").execute()
        remaining_docs = supabase.table("rag_documents").select("count", count="exact").execute()
        logger.info(f"   Verification: {remaining_chunks.count} chunks, {remaining_docs.count} documents remaining")
        
    except Exception as e:
        logger.error(f"   ✗ Error clearing Supabase: {e}")
    
    # 2. Clear Qdrant
    logger.info("\n2. CLEARING QDRANT...")
    try:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        collection_name = "rag_visualizer_chunks"
        
        # Check if collection exists
        collections = qdrant.get_collections().collections
        if any(c.name == collection_name for c in collections):
            logger.info(f"   Deleting collection '{collection_name}'...")
            qdrant.delete_collection(collection_name=collection_name)
            logger.info(f"   ✓ Deleted Qdrant collection '{collection_name}'")
            
            # Recreate empty collection
            from qdrant_client.models import Distance, VectorParams
            logger.info(f"   Recreating empty collection '{collection_name}'...")
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,  # text-embedding-3-small dimension
                    distance=Distance.COSINE
                )
            )
            logger.info(f"   ✓ Recreated empty Qdrant collection")
        else:
            logger.info(f"   Collection '{collection_name}' does not exist")
            
    except Exception as e:
        logger.error(f"   ✗ Error clearing Qdrant: {e}")
    
    # 3. Clear Neo4j
    logger.info("\n3. CLEARING NEO4J...")
    try:
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        with driver.session() as session:
            # Count nodes before deletion
            count_result = session.run("MATCH (n) RETURN count(n) as count")
            initial_count = count_result.single()["count"]
            logger.info(f"   Found {initial_count} nodes to delete...")
            
            # Delete all nodes and relationships
            logger.info("   Deleting all nodes and relationships...")
            session.run("MATCH (n) DETACH DELETE n")
            logger.info(f"   ✓ Deleted all Neo4j nodes and relationships")
            
            # Verify deletion
            verify_result = session.run("MATCH (n) RETURN count(n) as count")
            final_count = verify_result.single()["count"]
            logger.info(f"   Verification: {final_count} nodes remaining")
            
        driver.close()
        
    except Exception as e:
        logger.error(f"   ✗ Error clearing Neo4j: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("DATA CLEARANCE PROCESS COMPLETED")
    logger.info("=" * 60)

if __name__ == "__main__":
    clear_all_data()