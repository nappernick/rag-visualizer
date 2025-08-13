from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


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


class Document(BaseModel):
    id: str
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    doc_type: Optional[str] = None  # 'project', 'values', 'meeting', etc.
    temporal_weight: Optional[float] = 1.0  # Pre-computed temporal relevance


class Chunk(BaseModel):
    id: str
    document_id: str
    content: str
    position: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at_ms: Optional[int] = None  # Epoch milliseconds for efficient filtering


class Entity(BaseModel):
    id: str
    text: str
    type: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Relationship(BaseModel):
    id: str
    source_id: str
    target_id: str
    type: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str = "vector"  # vector, keyword, metadata, graph, hybrid