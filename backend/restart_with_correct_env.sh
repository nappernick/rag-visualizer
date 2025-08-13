#!/bin/bash
# Restart backend with correct environment variables

echo "🔄 Restarting RAG Visualizer with correct environment..."

# Export correct environment variables
export NEO4J_URI="neo4j+s://c7e3cd4b.databases.neo4j.io"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="tQciHNp_L6fd5Op0qLFT4A_n1Z0uSd8HffzNby94AIA"
export NEO4J_DATABASE="neo4j"

export QDRANT_URL="https://a2d4af28-8c4d-447f-98d6-d845c4b48c40.us-west-2-0.aws.cloud.qdrant.io"
export QDRANT_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ipo8g6LEMtrW7hiJWV8lzgXa8c1Zn1C-Ae-xYE0jZnw"

export OPENAI_API_KEY="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"
export EMBEDDING_MODEL="openai"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"

# Stop existing processes
cd /home/nmatnich/rag-visualizer
./stop.sh

# Start with new environment
./start.sh

echo "✅ Restarted with correct environment variables"