#!/usr/bin/env python3
"""Direct Neo4j connection test"""

from neo4j import GraphDatabase

# Use the exact credentials you provided
URI = "neo4j+s://c7e3cd4b.databases.neo4j.io"
AUTH = ("neo4j", "tQciHNp_L6fd5Op0qLFT4A_n1Z0uSd8HffzNby94AIA")

print(f"Attempting to connect to: {URI}")

try:
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        print("✅ Neo4j connection successful!")
        
        # Try a simple query
        with driver.session() as session:
            result = session.run("RETURN 1 AS num")
            record = result.single()
            print(f"Test query result: {record['num']}")
            
except Exception as e:
    print(f"❌ Connection failed: {e}")
    import traceback
    traceback.print_exc()