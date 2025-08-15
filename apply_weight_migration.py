#!/usr/bin/env python3
"""
Apply weight system migration to Supabase database
"""
import os
import asyncio
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

async def apply_migration():
    # Get Supabase credentials
    url = os.getenv('SUPABASE_URL')
    service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not url or not service_key:
        print("Error: Missing Supabase credentials")
        return False
    
    print(f"Connecting to Supabase at: {url}")
    
    # Create client with service role key for full access
    client = create_client(url, service_key)
    
    # Read the migration SQL
    migration_file = '/home/nmatnich/rag-visualizer/migrations/add_document_weights.sql'
    with open(migration_file, 'r') as f:
        migration_sql = f.read()
    
    # Split the migration into individual statements
    # Remove comments and split by semicolons
    statements = []
    current_statement = []
    
    for line in migration_sql.split('\n'):
        # Skip comment lines
        if line.strip().startswith('--'):
            continue
        
        current_statement.append(line)
        
        # Check if this line ends a statement
        if line.strip().endswith(';'):
            stmt = '\n'.join(current_statement).strip()
            if stmt:
                statements.append(stmt)
            current_statement = []
    
    print(f"Found {len(statements)} SQL statements to execute")
    
    # Execute each statement
    success_count = 0
    for i, stmt in enumerate(statements, 1):
        # Get first 50 chars of statement for logging
        stmt_preview = stmt[:50].replace('\n', ' ')
        if len(stmt) > 50:
            stmt_preview += '...'
        
        print(f"\nStatement {i}/{len(statements)}: {stmt_preview}")
        
        try:
            # Use RPC to execute raw SQL
            result = client.rpc('exec_sql', {'query': stmt}).execute()
            print(f"  ✓ Success")
            success_count += 1
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a "already exists" error which we can ignore
            if 'already exists' in error_msg.lower():
                print(f"  ⚠ Already exists (skipping)")
                success_count += 1
            else:
                print(f"  ✗ Error: {error_msg}")
    
    print(f"\n{'='*50}")
    print(f"Migration complete: {success_count}/{len(statements)} statements successful")
    
    return success_count == len(statements)

if __name__ == "__main__":
    success = asyncio.run(apply_migration())
    exit(0 if success else 1)