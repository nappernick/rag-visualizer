#!/usr/bin/env python3
"""
Comprehensive Integration Test for RAG Visualizer
Tests all external service connections and document processing flow
"""

import os
import sys
import json
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ANSI color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_section(title: str):
    """Print a section header"""
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}{title}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}")

def print_success(message: str):
    """Print success message"""
    print(f"{GREEN}✅ {message}{RESET}")

def print_error(message: str):
    """Print error message"""
    print(f"{RED}❌ {message}{RESET}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{YELLOW}⚠️  {message}{RESET}")

def print_info(message: str):
    """Print info message"""
    print(f"{BLUE}ℹ️  {message}{RESET}")

class IntegrationTester:
    """Test all RAG Visualizer integrations"""
    
    def __init__(self):
        self.results = {}
        self.backend_url = "http://localhost:8745"
    
    async def test_backend_health(self) -> bool:
        """Test if backend is running"""
        print_section("Backend Health Check")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.backend_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        print_success(f"Backend is running: {data}")
                        return True
                    else:
                        print_error(f"Backend returned status {response.status}")
                        return False
        except Exception as e:
            print_error(f"Backend not accessible: {e}")
            return False
    
    def test_supabase(self) -> bool:
        """Test Supabase connection"""
        print_section("Supabase Database Test")
        try:
            from supabase import create_client
            
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_ANON_API_KEY")
            
            if not url or not key:
                print_error("Supabase credentials not found in .env")
                return False
            
            print_info(f"Connecting to Supabase at: {url[:30]}...")
            client = create_client(url, key)
            
            # Test creating tables if they don't exist
            tables_to_check = ["documents", "chunks", "entities", "relationships"]
            
            for table_name in tables_to_check:
                try:
                    # Try to select from table
                    response = client.table(table_name).select("*").limit(1).execute()
                    print_success(f"Table '{table_name}' accessible")
                except Exception as e:
                    print_warning(f"Table '{table_name}' might not exist: {e}")
            
            print_success("Supabase connection successful")
            return True
            
        except Exception as e:
            print_error(f"Supabase connection failed: {e}")
            return False
    
    def test_qdrant(self) -> bool:
        """Test Qdrant connection"""
        print_section("Qdrant Vector Database Test")
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            url = os.getenv("QDRANT_URL")
            api_key = os.getenv("QDRANT_API_KEY")
            
            if not url or not api_key:
                print_error("Qdrant credentials not found in .env")
                return False
            
            print_info(f"Connecting to Qdrant at: {url[:50]}...")
            
            # Create client with longer timeout
            client = QdrantClient(
                url=url,
                api_key=api_key,
                timeout=60  # 60 second timeout
            )
            
            # Get collections
            collections = client.get_collections()
            print_success(f"Connected! Found {len(collections.collections)} collections")
            
            for collection in collections.collections:
                print_info(f"  - Collection: {collection.name}")
            
            # Try to create or update the main collection
            collection_name = os.getenv("QDRANT_COLLECTION", "chunks")
            embed_dim = int(os.getenv("EMBED_DIM", "1536"))
            
            try:
                # Check if collection exists
                collection_info = client.get_collection(collection_name)
                print_success(f"Collection '{collection_name}' exists with {collection_info.points_count} points")
            except:
                # Create collection if it doesn't exist
                print_info(f"Creating collection '{collection_name}'...")
                client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embed_dim,
                        distance=Distance.COSINE
                    )
                )
                print_success(f"Created collection '{collection_name}'")
            
            return True
            
        except Exception as e:
            print_error(f"Qdrant connection failed: {e}")
            return False
    
    def test_neo4j(self) -> bool:
        """Test Neo4j connection"""
        print_section("Neo4j Graph Database Test")
        try:
            from neo4j import GraphDatabase
            
            uri = os.getenv("NEO4J_URI")
            user = os.getenv("NEO4J_USER")
            password = os.getenv("NEO4J_PASSWORD")
            
            if not uri or not user or not password:
                print_error("Neo4j credentials not found in .env")
                return False
            
            print_info(f"Connecting to Neo4j at: {uri}")
            
            # Try connection with longer timeout
            driver = GraphDatabase.driver(
                uri, 
                auth=(user, password),
                connection_timeout=30,
                max_connection_pool_size=10
            )
            
            # Verify connectivity
            driver.verify_connectivity()
            print_success("Neo4j connection verified")
            
            # Run a test query
            with driver.session() as session:
                result = session.run("RETURN 1 AS test")
                value = result.single()["test"]
                if value == 1:
                    print_success("Test query successful")
            
            # Get database info
            with driver.session() as session:
                result = session.run("""
                    CALL dbms.components() 
                    YIELD name, versions 
                    RETURN name, versions[0] as version
                """)
                for record in result:
                    print_info(f"  - {record['name']}: {record['version']}")
            
            driver.close()
            return True
            
        except Exception as e:
            print_error(f"Neo4j connection failed: {e}")
            print_info("Note: Neo4j might need time to start up or DNS to resolve")
            return False
    
    def test_openai(self) -> bool:
        """Test OpenAI API"""
        print_section("OpenAI API Test")
        try:
            import openai
            
            api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                print_error("OpenAI API key not found in .env")
                return False
            
            # Create client
            client = openai.OpenAI(api_key=api_key)
            
            # Test embedding generation
            model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            print_info(f"Testing embedding model: {model}")
            
            response = client.embeddings.create(
                model=model,
                input="This is a test document for the RAG visualizer"
            )
            
            embedding_dim = len(response.data[0].embedding)
            print_success(f"Embedding generated successfully! Dimension: {embedding_dim}")
            
            # Verify dimension matches expected
            expected_dim = int(os.getenv("EMBED_DIM", "1536"))
            if embedding_dim == expected_dim:
                print_success(f"Embedding dimension matches expected: {expected_dim}")
            else:
                print_warning(f"Embedding dimension {embedding_dim} doesn't match expected {expected_dim}")
            
            return True
            
        except Exception as e:
            print_error(f"OpenAI API test failed: {e}")
            return False
    
    async def test_document_upload(self) -> bool:
        """Test complete document upload and processing flow"""
        print_section("Document Upload & Processing Test")
        
        try:
            # Prepare test document
            test_content = """
            # RAG Visualizer Test Document
            
            This is a test document for the RAG visualizer system.
            It contains information about retrieval-augmented generation (RAG).
            
            ## Key Concepts
            - Vector embeddings
            - Knowledge graphs
            - Semantic search
            - Document chunking
            
            ## Technologies
            We use OpenAI for embeddings, Qdrant for vector storage,
            Neo4j for graph database, and Supabase for document storage.
            """
            
            # Create form data
            data = aiohttp.FormData()
            data.add_field('file',
                          test_content.encode('utf-8'),
                          filename='test_document.md',
                          content_type='text/markdown')
            
            async with aiohttp.ClientSession() as session:
                # Upload document
                print_info("Uploading test document...")
                async with session.post(f"{self.backend_url}/api/documents/upload", data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        doc_id = result.get('id')
                        print_success(f"Document uploaded! ID: {doc_id}")
                        
                        # Get document details
                        async with session.get(f"{self.backend_url}/api/documents/{doc_id}") as doc_response:
                            if doc_response.status == 200:
                                doc_data = await doc_response.json()
                                print_success(f"Document retrieved: {doc_data.get('title')}")
                        
                        # Get chunks
                        async with session.get(f"{self.backend_url}/api/documents/{doc_id}/chunks") as chunks_response:
                            if chunks_response.status == 200:
                                chunks = await chunks_response.json()
                                print_success(f"Found {len(chunks)} chunks")
                        
                        return True
                    else:
                        error_text = await response.text()
                        print_error(f"Upload failed ({response.status}): {error_text}")
                        return False
                        
        except Exception as e:
            print_error(f"Document upload test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all integration tests"""
        print_section("RAG Visualizer Integration Test Suite")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Test backend health first
        self.results["Backend"] = await self.test_backend_health()
        
        if not self.results["Backend"]:
            print_warning("Backend not running. Some tests will be skipped.")
        
        # Test external services
        self.results["Supabase"] = self.test_supabase()
        self.results["Qdrant"] = self.test_qdrant()
        self.results["Neo4j"] = self.test_neo4j()
        self.results["OpenAI"] = self.test_openai()
        
        # Test document flow if backend is running
        if self.results["Backend"]:
            self.results["Document Upload"] = await self.test_document_upload()
        
        # Print summary
        print_section("Test Results Summary")
        
        all_passed = True
        for service, passed in self.results.items():
            if passed:
                print_success(f"{service}: PASSED")
            else:
                print_error(f"{service}: FAILED")
                all_passed = False
        
        print()
        if all_passed:
            print_success("🎉 All tests passed!")
        else:
            print_warning("Some tests failed. Check the output above for details.")
        
        # Print configuration tips
        if not self.results.get("Neo4j", False):
            print_info("\nNeo4j Tips:")
            print_info("- Ensure Neo4j instance is running and accessible")
            print_info("- Check if DNS can resolve the Neo4j host")
            print_info("- Verify credentials are correct")
            print_info("- Neo4j may need time to start up (wait a few minutes)")
        
        if not self.results.get("Qdrant", False):
            print_info("\nQdrant Tips:")
            print_info("- Check if Qdrant cloud instance is active")
            print_info("- Verify API key is correct")
            print_info("- Ensure network allows HTTPS connections to Qdrant")
        
        return all_passed

async def main():
    """Main test runner"""
    tester = IntegrationTester()
    success = await tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)