#!/usr/bin/env python3
"""
Clean Neo4j database completely
"""
import os
import sys
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

def clean_neo4j():
    """Completely clean Neo4j database"""
    
    # Get Neo4j credentials
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not uri or not password:
        print("❌ Neo4j credentials not found in environment")
        return False
    
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session() as session:
            # Count before deletion
            result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = result.single()["count"]
            
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()["count"]
            
            print(f"📊 Found {node_count} nodes and {rel_count} relationships")
            
            if node_count == 0 and rel_count == 0:
                print("✅ Neo4j is already clean")
                return True
            
            # Delete everything
            print("🗑️  Deleting all nodes and relationships...")
            session.run("MATCH (n) DETACH DELETE n")
            
            # Verify deletion
            result = session.run("MATCH (n) RETURN count(n) as count")
            remaining = result.single()["count"]
            
            if remaining == 0:
                print(f"✅ Successfully deleted {node_count} nodes and {rel_count} relationships")
                return True
            else:
                print(f"⚠️  Warning: {remaining} nodes still remain")
                
                # Try more aggressive deletion
                print("🔄 Attempting more aggressive cleanup...")
                
                # Delete in batches to avoid memory issues
                batch_size = 1000
                while True:
                    result = session.run(f"""
                        MATCH (n)
                        WITH n LIMIT {batch_size}
                        DETACH DELETE n
                        RETURN count(n) as deleted
                    """)
                    deleted = result.single()["deleted"]
                    if deleted == 0:
                        break
                    print(f"  Deleted batch of {deleted} nodes")
                
                # Final check
                result = session.run("MATCH (n) RETURN count(n) as count")
                final_count = result.single()["count"]
                
                if final_count == 0:
                    print("✅ Neo4j fully cleaned after aggressive cleanup")
                    return True
                else:
                    print(f"❌ Failed to clean Neo4j - {final_count} nodes remain")
                    
                    # Show what remains
                    result = session.run("""
                        MATCH (n)
                        RETURN labels(n) as labels, count(n) as count
                        ORDER BY count DESC
                        LIMIT 10
                    """)
                    print("\nRemaining node types:")
                    for record in result:
                        print(f"  {record['labels']}: {record['count']}")
                    
                    return False
        
        driver.close()
        return True
        
    except Exception as e:
        print(f"❌ Error cleaning Neo4j: {e}")
        return False

if __name__ == "__main__":
    success = clean_neo4j()
    sys.exit(0 if success else 1)