# RAG Visualizer

A comprehensive visualization system for understanding and debugging Retrieval-Augmented Generation (RAG) pipelines. This project provides interactive visualizations for document chunking, knowledge graph construction, and retrieval processes.

## 📁 Project Structure

```
rag-visualizer/
├── backend/                 # FastAPI backend service
│   ├── src/
│   │   ├── api/            # REST API endpoints
│   │   ├── core/           # Core RAG logic
│   │   │   ├── chunking/   # Document chunking strategies
│   │   │   ├── embedding/  # Embedding generation
│   │   │   ├── retrieval/  # Retrieval strategies
│   │   │   ├── graph/      # Knowledge graph extraction
│   │   │   └── temporal/   # Temporal processing
│   │   ├── services/       # Service layer
│   │   ├── storage/        # Database adapters
│   │   └── models/         # Pydantic models
│   └── requirements.txt    # Python dependencies
├── frontend/               # React TypeScript frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API client services
│   │   └── types/          # TypeScript definitions
│   ├── package.json        # Node dependencies
│   └── nixpacks.toml       # Railway deployment config
├── scripts/                # Utility and management scripts
│   ├── start.sh           # Start development servers
│   ├── stop.sh            # Stop development servers
│   ├── force_restart.sh   # Force restart services
│   ├── create_tables.py   # Database initialization
│   └── setup_storage.py   # Storage setup
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── test_*.py          # Test files
├── migrations/            # Database migrations
│   ├── *.sql             # SQL migration files
│   └── *.cypher          # Neo4j schema files
├── Dockerfile            # Backend Docker configuration
├── railway.json          # Railway deployment config
└── docker-compose.yml    # Local development infrastructure
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ with pip
- Node.js 18+ with npm or bun
- Docker and Docker Compose
- Git

### One-Command Start

```bash
# Start everything (backend + frontend)
./start.sh

# Stop everything
./stop.sh
```

The start script will:
- ✅ Start backend on http://localhost:8642
- ✅ Start frontend on http://localhost:5892
- ✅ Create log files for debugging
- ✅ Handle virtual environment activation

## 📜 Scripts

All management scripts are located in the `scripts/` directory:

### Development Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `start.sh` | Start both backend and frontend | `./scripts/start.sh` or `./start.sh` |
| `stop.sh` | Stop all services cleanly | `./scripts/stop.sh` or `./stop.sh` |
| `force_restart.sh` | Force restart with clean state | `./scripts/force_restart.sh` |

### Setup Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `create_tables.py` | Initialize database tables | `python scripts/create_tables.py` |
| `setup_storage.py` | Configure storage backends | `python scripts/setup_storage.py` |
| `run_tests.py` | Run test suite | `python scripts/run_tests.py` |

## 🧪 Testing

### Test Coverage

Our test suite covers:

- **Unit Tests** (`tests/unit/`)
  - Date extraction and temporal processing
  - Query enhancement logic
  - Hybrid search algorithms
  - Graph RAG functionality

- **Integration Tests** (`tests/integration/`)
  - Fusion controller operations
  - End-to-end RAG pipeline
  - Service integrations

- **System Tests** (`tests/`)
  - Complete system functionality
  - Neo4j and Qdrant connections
  - Supabase integration
  - UI functionality testing

### Running Tests

```bash
# Run all tests
python scripts/run_tests.py

# Run specific test file
python -m pytest tests/test_services.py

# Run with coverage
python -m pytest --cov=backend/src tests/

# Run integration tests only
python -m pytest tests/integration/
```

## 🏗️ Architecture

### Backend (FastAPI)

The backend is a FastAPI application with:

- **API Layer**: RESTful endpoints for all operations
- **Service Layer**: Business logic and orchestration
- **Storage Layer**: Abstractions for PostgreSQL, Qdrant, Neo4j
- **Core Components**:
  - Document processing and chunking
  - Entity and relationship extraction
  - Vector and graph-based retrieval
  - Temporal analysis
  - Query enhancement

### Frontend (React + TypeScript)

The frontend provides:

- **Interactive Visualizations**: D3.js and Cytoscape.js powered
- **Real-time Updates**: WebSocket support for live data
- **Component Library**:
  - ChunkVisualizer: Document chunk exploration
  - GraphViewer: Knowledge graph visualization
  - FusionControls: Retrieval strategy controls

## 🚢 Deployment

### Railway Deployment

The project is configured for Railway deployment with:

1. **Backend Service** (Dockerfile)
   - Python 3.11 slim base image
   - Automatic spaCy model download
   - Environment variable support
   - Health check endpoint at `/health`

2. **Frontend Service** (nixpacks.toml)
   - Bun package manager for speed
   - Production build with Vite
   - Served with `serve` package
   - Port configuration via environment

### Environment Variables

Create `.env` file (see `.env.example`):

```bash
# Backend
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Frontend (for Railway)
VITE_API_BASE=https://your-backend.railway.app
```

### Deployment Steps

1. **Push to GitHub**
   ```bash
   git add -A
   git commit -m "Deploy to Railway"
   git push origin main
   ```

2. **Railway Setup**
   - Create new project on Railway
   - Add backend service (point to Dockerfile)
   - Add frontend service (point to frontend/ directory)
   - Configure environment variables
   - Generate public domains for both services

3. **Configure Frontend**
   - Set `VITE_API_BASE` to backend's public URL
   - Set port to 8080 in Railway networking settings

## 🔧 Development

### Backend Development

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Start development server
python -m uvicorn src.main:app --reload --port 8642
```

### Frontend Development

```bash
cd frontend

# Install dependencies (using bun for speed)
bun install
# OR using npm
npm install

# Start development server
bun run dev
# OR
npm run dev
```

### Database Setup

```bash
# Start local databases
docker-compose up -d

# Initialize tables
python scripts/create_tables.py

# Setup storage backends
python scripts/setup_storage.py
```

## 📊 Features

### Document Processing & Chunking
- **Multiple Strategies**: Standard, Hierarchical, and Semantic chunking
- **Visual Explorer**: Interactive chunk hierarchy visualization
- **Token Analysis**: Real-time token counting and overlap

### Knowledge Graph Construction
- **Entity Extraction**: NER and pattern-based recognition
- **Relationship Discovery**: Automatic relationship extraction
- **Graph Visualization**: Interactive force-directed graphs
- **Community Detection**: Visual clustering of entities

### Retrieval Pipeline
- **Hybrid Retrieval**: Vector + graph-based approaches
- **Query Analysis**: Real-time intent classification
- **Reranking**: Cross-encoder reranking visualization
- **Score Distribution**: Interactive retrieval score charts

### Temporal Analysis
- **Date Extraction**: Automatic temporal entity detection
- **Time-aware Retrieval**: Temporal relevance scoring
- **Decay Functions**: Configurable temporal decay curves

## 🔍 API Endpoints

### Document Management
- `POST /api/documents` - Create document
- `POST /api/documents/upload` - Upload file
- `GET /api/documents` - List documents
- `GET /api/documents/{id}` - Get document

### Chunking
- `POST /api/chunking` - Chunk document
- `GET /api/documents/{id}/chunks` - Get chunks

### Graph Operations
- `POST /api/graph/extract` - Extract entities
- `GET /api/graph/{id}/entities` - Get entities
- `GET /api/graph/{id}/relationships` - Get relationships

### Query & Retrieval
- `POST /api/query` - Query documents
- `POST /api/fusion/search` - Fusion search

### Visualization
- `GET /api/visualization/{id}` - Get visualization data

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Force stop all services
   ./scripts/force_restart.sh
   ./scripts/stop.sh
   ```

2. **Backend Won't Start**
   ```bash
   # Check logs
   tail -f backend.log
   
   # Verify virtual environment
   source venv/bin/activate
   pip install -r backend/requirements.txt
   ```

3. **Frontend Build Fails**
   ```bash
   # Clear cache and reinstall
   cd frontend
   rm -rf node_modules dist
   bun install
   bun run build
   ```

4. **Database Connection Issues**
   ```bash
   # Restart Docker containers
   docker-compose down
   docker-compose up -d
   
   # Check container status
   docker ps
   ```

### Log Files

- **Backend**: `backend.log`
- **Frontend**: `frontend.log`
- **Startup Debug**: `startup_debug.log`

Monitor logs in real-time:
```bash
tail -f backend.log
tail -f frontend.log
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add/update tests as needed
5. Run test suite (`python scripts/run_tests.py`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Based on SmartRAG research
- Powered by FastAPI and React
- Visualization with D3.js and Cytoscape.js
- NLP processing with spaCy
- Vector search with Qdrant
- Graph database with Neo4j