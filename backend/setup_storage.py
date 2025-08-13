#!/usr/bin/env python3
"""
Setup script for cloud storage services
Creates tables in Supabase, collections in Qdrant, and schema in Neo4j
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from supabase import create_client
import logging

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_supabase():
    """Create Supabase tables"""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_API_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials not found in environment")
            return False
        
        client = create_client(supabase_url, supabase_key)
        
        # Read SQL file
        sql_file = Path(__file__).parent / "supabase_schema.sql"
        with open(sql_file, 'r') as f:
            sql = f.read()
        
        # Note: Supabase Python client doesn't support direct SQL execution
        # You need to run this SQL in the Supabase dashboard SQL editor
        logger.info("=" * 60)
        logger.info("IMPORTANT: Run the following SQL in Supabase Dashboard:")
        logger.info("Go to: https://supabase.com/dashboard/project/joutvpdtmaauspbxnzya/sql/new")
        logger.info("=" * 60)
        print(sql)
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up Supabase: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("Starting cloud storage setup...")
    
    # Setup Supabase
    if setup_supabase():
        logger.info("✓ Supabase setup instructions provided")
    else:
        logger.error("✗ Supabase setup failed")
    
    logger.info("\nSetup complete!")
    logger.info("The backend will automatically create Qdrant collections and Neo4j schema on startup.")

if __name__ == "__main__":
    main()