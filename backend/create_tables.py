#!/usr/bin/env python3
"""
Create Supabase tables using the Supabase Python client
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
import psycopg2
from urllib.parse import urlparse
import logging

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_tables_via_postgres():
    """Create tables using direct PostgreSQL connection"""
    try:
        # Parse Supabase PostgreSQL URI
        db_uri = os.getenv("SUPABASE_URI")
        if not db_uri:
            logger.error("SUPABASE_URI not found in environment")
            return False
        
        # Connect to PostgreSQL with SSL
        # Add sslmode if not present
        if "sslmode=" not in db_uri:
            db_uri += "?sslmode=require"
        conn = psycopg2.connect(db_uri)
        cur = conn.cursor()
        
        logger.info("Connected to Supabase PostgreSQL")
        
        # Create documents table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT,
                doc_type TEXT,
                status TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        logger.info("Created rag_documents table")
        
        # Create chunks table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT REFERENCES rag_documents(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                chunk_index INTEGER,
                chunk_type TEXT,
                tokens INTEGER,
                parent_id TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        logger.info("Created rag_chunks table")
        
        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_rag_documents_status ON rag_documents(status)",
            "CREATE INDEX IF NOT EXISTS idx_rag_documents_created_at ON rag_documents(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_id ON rag_chunks(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_rag_chunks_parent_id ON rag_chunks(parent_id)",
            "CREATE INDEX IF NOT EXISTS idx_rag_chunks_chunk_index ON rag_chunks(chunk_index)"
        ]
        
        for index_sql in indexes:
            cur.execute(index_sql)
            logger.info(f"Created index: {index_sql.split(' ')[-1]}")
        
        # Create updated_at trigger
        cur.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql'
        """)
        
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger 
                    WHERE tgname = 'update_rag_documents_updated_at'
                ) THEN
                    CREATE TRIGGER update_rag_documents_updated_at 
                    BEFORE UPDATE ON rag_documents 
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
                END IF;
            END $$
        """)
        logger.info("Created triggers")
        
        # Commit changes
        conn.commit()
        cur.close()
        conn.close()
        
        logger.info("✅ Successfully created all tables and indexes in Supabase")
        return True
        
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        return False

def verify_tables():
    """Verify tables were created"""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_API_KEY")
        
        client = create_client(supabase_url, supabase_key)
        
        # Try to query the tables
        docs = client.table("rag_documents").select("*").limit(1).execute()
        logger.info("✅ rag_documents table exists and is accessible")
        
        chunks = client.table("rag_chunks").select("*").limit(1).execute()
        logger.info("✅ rag_chunks table exists and is accessible")
        
        return True
        
    except Exception as e:
        logger.error(f"Error verifying tables: {e}")
        return False

def main():
    """Main function"""
    logger.info("Starting Supabase table creation...")
    
    # Create tables
    if create_tables_via_postgres():
        logger.info("Tables created successfully")
        
        # Verify
        if verify_tables():
            logger.info("✅ All tables verified and ready to use!")
        else:
            logger.warning("⚠️  Tables created but verification failed")
    else:
        logger.error("❌ Failed to create tables")

if __name__ == "__main__":
    main()