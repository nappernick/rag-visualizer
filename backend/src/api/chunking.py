"""
Chunking API endpoints for document processing
"""
from typing import List, Optional, Dict
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
import hashlib
from datetime import datetime

from ..db import get_session
from ..core.chunking.semantic_chunker import SemanticChunker
from ..core.chunking.hierarchical import HierarchicalChunker
from ..core.chunking.base import StandardChunker
from ..models import Document as DocumentSchema
from ..services.storage import get_storage_service
from ..services.vector_service import get_vector_service
from ..services.embedding_service import get_embedding_service

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
async def chunk_document(request: ChunkingRequest, 
                        db: Session = Depends(get_session), 
                        storage=Depends(get_storage_service),
                        embedding_service=Depends(get_embedding_service),
                        vector_service=Depends(get_vector_service)):
    """Chunk a document using various strategies."""
    
    start_time = datetime.now()
    chunks = []
    
    # Create a mock document object for the chunkers
    mock_document = DocumentSchema(
        id=request.document_id,
        title="Uploaded Document",
        content=request.content
    )
    
    # Choose chunker based on strategy
    if request.strategy == "hierarchical":
        chunker = HierarchicalChunker(
            max_chunk_size=request.max_chunk_size,
            chunk_overlap=request.chunk_overlap,
            create_summaries=True
        )
        chunk_objects = chunker.chunk_document(mock_document)
    elif request.strategy == "semantic":
        # Use semantic chunker if available, otherwise standard
        try:
            chunker = SemanticChunker(
                chunk_size=request.max_chunk_size,
                chunk_overlap=request.chunk_overlap
            )
            chunk_texts = chunker.chunk(request.content)
            # Convert to Chunk objects
            chunk_objects = []
            for i, text in enumerate(chunk_texts):
                chunk_id = hashlib.md5(f"{request.document_id}_{i}_{text[:50]}".encode()).hexdigest()[:12]
                chunk_obj = type('Chunk', (), {
                    'id': chunk_id,
                    'content': text,
                    'document_id': request.document_id,
                    'chunk_index': i,
                    'chunk_type': 'standard',
                    'tokens': len(text.split()),
                    'metadata': {'method': 'semantic'},
                    'parent_id': None,
                    'children_ids': [],
                    'created_at': datetime.now().isoformat()
                })()
                chunk_objects.append(chunk_obj)
        except Exception as e:
            # Fall back to standard chunking
            chunker = StandardChunker(
                max_chunk_size=request.max_chunk_size,
                chunk_overlap=request.chunk_overlap
            )
            chunk_objects = chunker.chunk_document(mock_document)
    else:
        # Standard chunking
        chunker = StandardChunker(
            max_chunk_size=request.max_chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        chunk_objects = chunker.chunk_document(mock_document)
    
    # Convert chunk objects to dictionaries and response objects
    chunk_dicts = []
    for chunk_obj in chunk_objects:
        # Handle chunk object attributes  
        chunk_id = getattr(chunk_obj, 'id', hashlib.md5(f"{request.document_id}_{len(chunk_dicts)}".encode()).hexdigest()[:12])
        chunk_content = getattr(chunk_obj, 'content', '')
        chunk_index = getattr(chunk_obj, 'chunk_index', len(chunk_dicts))
        chunk_type = getattr(chunk_obj, 'chunk_type', 'standard')
        tokens = getattr(chunk_obj, 'tokens', len(chunk_content.split()))
        metadata = getattr(chunk_obj, 'metadata', {})
        parent_id = getattr(chunk_obj, 'parent_id', None)
        children_ids = getattr(chunk_obj, 'children_ids', [])
        created_at_raw = getattr(chunk_obj, 'created_at', datetime.now())
        created_at = created_at_raw.isoformat() if isinstance(created_at_raw, datetime) else str(created_at_raw)
        
        # Ensure metadata includes strategy info
        metadata.update({
            "method": request.strategy,
            "chunk_size": request.max_chunk_size,
            "overlap": request.chunk_overlap
        })
        
        chunk_dict = {
            "id": chunk_id,
            "document_id": request.document_id,
            "content": chunk_content,
            "chunk_index": chunk_index,
            "chunk_type": chunk_type,
            "tokens": tokens,
            "metadata": metadata,
            "parent_id": parent_id,
            "children_ids": children_ids,
            "created_at": created_at
        }
        
        chunk = ChunkResponse(
            id=chunk_id,
            document_id=request.document_id,
            content=chunk_content,
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            tokens=tokens,
            metadata=metadata,
            parent_id=parent_id,
            children_ids=children_ids,
            created_at=created_at
        )
        
        chunks.append(chunk)
        chunk_dicts.append(chunk_dict)
    
    # Store chunks in Supabase
    if chunk_dicts:
        await storage.store_chunks(chunk_dicts)
    
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