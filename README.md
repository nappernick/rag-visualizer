# RAG Visualizer

A comprehensive visualization system for understanding and debugging Retrieval-Augmented Generation (RAG) pipelines. This project provides interactive visualizations for document chunking, knowledge graph construction, and retrieval processes.

## Features

### 🔍 Document Processing & Chunking
- **Multiple Chunking Strategies**: Standard, Hierarchical, and Semantic chunking
- **Visual Chunk Explorer**: Interactive visualization of chunk hierarchy, relationships, and metadata
- **Token Analysis**: Real-time token counting and overlap visualization

### 🕸️ Knowledge Graph Construction
- **Entity Extraction**: NER-based and pattern-based entity recognition
- **Relationship Discovery**: Automatic relationship extraction between entities
- **Graph Visualization**: Interactive force-directed graph with D3.js
- **Community Detection**: Visual clustering of related entities

### 🎯 Retrieval Pipeline Visualization
- **Hybrid Retrieval**: Visual comparison of vector vs graph-based retrieval
- **Query Analysis**: Real-time query intent classification
- **Reranking Visualization**: See how cross-encoder reranking affects results
- **Score Distribution**: Interactive charts showing retrieval scores

### 📊 Embedding Space Explorer
- **2D/3D Projections**: Visualize chunk embeddings in reduced dimensions
- **Cluster Analysis**: Identify semantic clusters in your documents
- **Query Positioning**: See where queries land in embedding space

## Architecture

```
rag-visualizer/
├── backend/              # FastAPI backend
│   ├── src/
│   │   ├── api/         # REST endpoints
│   │   ├── core/        # Core RAG logic
│   │   │   ├── chunking/    # Document chunking strategies
│   │   │   ├── embedding/   # Embedding generation
│   │   │   ├── retrieval/   # Retrieval strategies
│   │   │   └── graph/       # Knowledge graph extraction
│   │   ├── storage/     # Database adapters
│   │   └── models/      # Pydantic models
│   └── config.yaml      # Configuration
├── frontend/            # Vite TypeScript React
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── services/    # API services
│   │   └── types/       # TypeScript definitions
│   └── package.json
└── docker-compose.yml   # Infrastructure setup
```

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ and npm
- Python 3.10+

### 1. Start Infrastructure

```bash
# Start databases (PostgreSQL, Qdrant, Neo4j, Redis)
docker-compose up -d
```

### 2. Start Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (for NER)
python -m spacy download en_core_web_sm

# Start FastAPI server
cd src
python main.py
```

The backend will be available at http://localhost:8000

### 3. Start Frontend

```bash
cd frontend

# Install dependencies
npm install

# Configure Tailwind CSS
npx tailwindcss init -p

# Start development server
npm run dev
```

The frontend will be available at http://localhost:5173

## Usage

### Document Processing

1. **Upload Document**: Upload a text, PDF, or markdown document
2. **Select Chunking Strategy**: Choose between Standard, Hierarchical, or Semantic
3. **Visualize Chunks**: Explore the chunk hierarchy, token distribution, and relationships

### Knowledge Graph

1. **Extract Entities**: Automatically extract entities from chunks
2. **Build Graph**: Construct knowledge graph with relationships
3. **Explore Graph**: Interactive graph visualization with filtering and search

### Retrieval Testing

1. **Enter Query**: Type your query in the search box
2. **Select Strategy**: Choose vector, graph, or hybrid retrieval
3. **View Results**: See retrieval results with scores and explanations
4. **Compare Strategies**: Side-by-side comparison of different retrieval methods

## API Endpoints

### Document Management
- `POST /api/documents` - Create document
- `POST /api/documents/upload` - Upload document file
- `GET /api/documents` - List documents
- `GET /api/documents/{id}` - Get document details

### Chunking
- `POST /api/chunking` - Chunk document with strategy
- `GET /api/documents/{id}/chunks` - Get document chunks

### Graph Operations
- `POST /api/graph/extract` - Extract entities and relationships
- `GET /api/graph/{id}/entities` - Get document entities
- `GET /api/graph/{id}/relationships` - Get relationships

### Visualization
- `GET /api/visualization/{id}` - Get all visualization data

### Query
- `POST /api/query` - Query documents with retrieval strategy

## Configuration

Edit `backend/config.yaml` to configure:

- Chunking parameters (size, overlap, strategies)
- Embedding models (OpenAI, local)
- Storage backends (PostgreSQL, Qdrant, Neo4j)
- Retrieval settings (hybrid weights, reranking)
- Graph extraction options (NER, patterns)

## Key Components

### Chunking Strategies

1. **Standard Chunking**: Fixed-size chunks with configurable overlap
2. **Hierarchical Chunking**: Preserves document structure with parent-child relationships
3. **Semantic Chunking**: (Future) Chunks based on semantic boundaries

### Entity Extraction

- **spaCy NER**: Standard named entity recognition
- **Technical Patterns**: Custom patterns for technical entities (models, frameworks, databases)
- **Relationship Extraction**: Dependency parsing and co-occurrence analysis

### Retrieval Methods

- **Vector Retrieval**: Dense retrieval using embeddings (Qdrant)
- **Graph Retrieval**: Entity and relationship traversal (Neo4j)
- **Hybrid Retrieval**: Weighted combination with adaptive strategy selection

## Development

### Backend Development

```bash
# Run tests
pytest

# Format code
black src/
isort src/

# Type checking
mypy src/
```

### Frontend Development

```bash
# Run tests
npm test

# Build for production
npm run build

# Type checking
npm run type-check
```

## Deployment

### Using Docker

```bash
# Build and run all services
docker-compose -f docker-compose.prod.yml up --build
```

### Manual Deployment

1. Set up databases (PostgreSQL, Qdrant, Neo4j)
2. Configure environment variables
3. Deploy backend with Uvicorn/Gunicorn
4. Build and serve frontend with nginx

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License

## Acknowledgments

- Based on research from the SmartRAG system
- Uses spaCy for NLP processing
- Visualization powered by D3.js and Cytoscape.js
- Built with FastAPI and React