"""
Chunking API endpoints for document processing
"""
from typing import List, Optional, Dict
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
import hashlib

from ..db import get_session
from ..core.chunking.semantic_chunker import SemanticChunker
from ..services.storage import storage_service
from ..services.vector_service import vector_service
from ..services.embedding_service import embedding_service

router = APIRouter(prefix="/api", tags=["chunking"])


class ChunkingRequest(BaseModel):
    document_id: str
    content: str
    strategy: str = "semantic"  # Changed from chunk_method to strategy
    max_chunk_size: int = 500  # Changed from chunk_size to max_chunk_size
    chunk_overlap: int = 50
    embedding_model: Optional[str] = None


class ChunkResponse(BaseModel):
    id: str
    document_id: str
    content: str
    chunk_index: int  # Frontend expects chunk_index, not position
    chunk_type: str = "standard"  # Add chunk_type field
    tokens: int = 0  # Add tokens field
    metadata: Dict = {}
    parent_id: Optional[str] = None
    children_ids: List[str] = []
    created_at: str = ""


class ChunkingResponse(BaseModel):
    document_id: str
    chunks: List[ChunkResponse]
    total_chunks: int
    strategy_used: str = ""
    processing_time_ms: float = 0.0
    hierarchy_depth: Optional[int] = None


@router.post("/chunking", response_model=ChunkingResponse)
async def chunk_document(request: ChunkingRequest, db: Session = Depends(get_session)):
    """Chunk a document using various strategies."""
    
    start_time = datetime.now()
    chunks = []
    
    if request.strategy in ["semantic", "hierarchical"]:
        # Use semantic chunking for both semantic and hierarchical strategies
        chunker = SemanticChunker(
            chunk_size=request.max_chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        chunk_texts = chunker.chunk(request.content)
    else:
        # Simple fixed-size chunking for "standard" strategy
        chunk_texts = []
        text = request.content
        for i in range(0, len(text), request.max_chunk_size - request.chunk_overlap):
            chunk_texts.append(text[i:i + request.max_chunk_size])
    
    # Create chunk objects
    chunk_dicts = []
    for i, chunk_text in enumerate(chunk_texts):
        chunk_id = hashlib.md5(f"{request.document_id}_{i}_{chunk_text[:50]}".encode()).hexdigest()[:12]
        
        # Determine chunk type based on strategy
        chunk_type = "hierarchical" if request.strategy == "hierarchical" else "standard"
        
        chunk_dict = {
            "id": chunk_id,
            "document_id": request.document_id,
            "content": chunk_text,
            "chunk_index": i,
            "chunk_type": chunk_type,
            "tokens": len(chunk_text.split()),
            "metadata": {
                "method": request.strategy,
                "chunk_size": request.max_chunk_size,
                "overlap": request.chunk_overlap
            },
            "parent_id": None,
            "children_ids": [],
            "created_at": datetime.now().isoformat()
        }
        
        chunk = ChunkResponse(
            id=chunk_id,
            document_id=request.document_id,
            content=chunk_text,
            chunk_index=i,
            chunk_type=chunk_type,
            tokens=len(chunk_text.split()),
            metadata=chunk_dict["metadata"],
            parent_id=None,
            children_ids=[],
            created_at=chunk_dict["created_at"]
        )
        
        chunks.append(chunk)
        chunk_dicts.append(chunk_dict)
    
    # Store chunks in Supabase
    if chunk_dicts:
        await storage_service.store_chunks(chunk_dicts)
    
    # Generate embeddings and store in Qdrant
    if chunk_dicts:
        chunk_texts_for_embedding = [chunk["content"] for chunk in chunk_dicts]
        embeddings = await embedding_service.generate_embeddings(chunk_texts_for_embedding)
        
        if embeddings:
            await vector_service.store_vectors(chunk_dicts, embeddings)
    
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return ChunkingResponse(
        document_id=request.document_id,
        chunks=chunks,
        total_chunks=len(chunks),
        strategy_used=request.strategy,
        processing_time_ms=processing_time,
        hierarchy_depth=3 if request.strategy == "hierarchical" else None
    )