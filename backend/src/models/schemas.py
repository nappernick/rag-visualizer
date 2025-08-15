"""
Data models for RAG Visualizer
"""
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid


# ID mapping types and models
IDType = Literal[
    "document",
    "chunk",
    "vector", 
    "entity",
    "relationship",
    "graph_node",
    "graph_edge",
]


class IDLinkIn(BaseModel):
    a_type: IDType
    a_id: str
    b_type: IDType
    b_id: str
    relation: Optional[str] = Field(default=None, description="Optional relation label")
    bidirectional: bool = True


class IDLinkOut(BaseModel):
    id: int
    a_type: IDType
    a_id: str
    b_type: IDType
    b_id: str
    relation: Optional[str] = None
    created_at: datetime


class TraverseResponse(BaseModel):
    origin_type: IDType
    origin_id: str
    related: Dict[IDType, List[str]] = Field(default_factory=dict)


class IngestChunk(BaseModel):
    id: str
    vector_id: Optional[str] = None
    entity_ids: Optional[List[str]] = None
    parent_id: Optional[str] = None
    graph_node_ids: Optional[List[str]] = None


class IngestDocument(BaseModel):
    id: str
    chunk_ids: Optional[List[str]] = None
    entity_ids: Optional[List[str]] = None
    chunks: Optional[List[IngestChunk]] = None


class ChunkType(str, Enum):
    STANDARD = "standard"
    HIERARCHICAL = "hierarchical"
    SUMMARY = "summary"
    CODE = "code"
    TABLE = "table"
    SECTION = "section"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    doc_type: str = "text"
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    document_id: str
    chunk_index: int
    chunk_type: ChunkType = ChunkType.STANDARD
    tokens: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class Entity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    entity_type: str
    description: Optional[str] = None  # Entity description for context
    document_ids: List[str] = Field(default_factory=list)
    chunk_ids: List[str] = Field(default_factory=list)
    frequency: int = 1
    salience: float = 0.0  # Importance score (0-1)
    embedding: Optional[List[float]] = None  # Entity embedding for similarity
    attributes: Dict[str, Any] = Field(default_factory=dict)  # Structured attributes
    context_snippets: List[str] = Field(default_factory=list)  # Example contexts
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Relationship(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    description: Optional[str] = None  # Relationship description
    weight: float = 1.0  # Strength of relationship
    confidence: float = 1.0  # Extraction confidence (0-1)
    bidirectional: bool = False  # If relationship goes both ways
    temporal: Optional[str] = None  # Time context if applicable
    document_ids: List[str] = Field(default_factory=list)
    chunk_ids: List[str] = Field(default_factory=list)  # Which chunks mention this
    evidence: List[str] = Field(default_factory=list)  # Text evidence for relationship
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    source: str  # "vector", "graph", "hybrid"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    highlights: List[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str
    max_results: int = 10
    retrieval_strategy: str = "hybrid"  # "vector", "graph", "hybrid"
    include_metadata: bool = True
    rerank: bool = True


class QueryResponse(BaseModel):
    query: str
    results: List[RetrievalResult]
    total_results: int
    retrieval_strategy: str
    processing_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkingRequest(BaseModel):
    document_id: str
    content: str
    strategy: str = "hierarchical"
    max_chunk_size: int = 800
    chunk_overlap: int = 100


class ChunkingResponse(BaseModel):
    document_id: str
    chunks: List[Chunk]
    total_chunks: int
    strategy_used: str
    processing_time_ms: float
    hierarchy_depth: Optional[int] = None


class GraphExtractionRequest(BaseModel):
    document_id: str
    chunks: List[Chunk]
    extract_entities: bool = True
    extract_relationships: bool = True
    use_spacy: bool = True


class GraphExtractionResponse(BaseModel):
    document_id: str
    entities: List[Entity]
    relationships: List[Relationship]
    total_entities: int
    total_relationships: int
    processing_time_ms: float


class VisualizationData(BaseModel):
    """Data structure for frontend visualizations"""
    
    class ChunkHierarchy(BaseModel):
        nodes: List[Dict[str, Any]]
        edges: List[Dict[str, Any]]
    
    class GraphData(BaseModel):
        nodes: List[Dict[str, Any]]
        edges: List[Dict[str, Any]]
        communities: Optional[List[List[str]]] = None
    
    class EmbeddingSpace(BaseModel):
        points: List[Dict[str, Any]]  # {id, x, y, z, label, cluster}
        clusters: Optional[List[Dict[str, Any]]] = None
    
    chunk_hierarchy: Optional[ChunkHierarchy] = None
    knowledge_graph: Optional[GraphData] = None
    embedding_space: Optional[EmbeddingSpace] = None
    retrieval_flow: Optional[Dict[str, Any]] = None