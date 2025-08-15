"""
Document management API endpoints
"""
from typing import List, Optional, Dict
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
import json
import hashlib
from datetime import datetime
import time

from ..db import get_session
from ..models import Document, Chunk
from ..core.temporal.temporal_utils import enrich_with_temporal_metadata
from ..core.temporal.date_extractor import extract_temporal_metadata
from ..services.storage import get_storage_service
from ..core.chunking.semantic_chunker import SemanticChunker
from ..core.chunking.hierarchical_chunker import HierarchicalChunker, HierarchicalChunk
from ..core.chunking.base import StandardChunker
from ..services.entity_service import get_entity_service
from ..core.service_manager import get_embedding_service, get_vector_service
import logging

logger = logging.getLogger(__name__)

# Try to import Textract processor
try:
    from ..core.ocr.textract_processor import TextractProcessor
    TEXTRACT_AVAILABLE = True
except ImportError:
    TextractProcessor = None
    TEXTRACT_AVAILABLE = False

router = APIRouter(prefix="/api/documents", tags=["documents"])


class DocumentResponse(BaseModel):
    id: str
    title: str
    content: Optional[str] = None
    created_at: Optional[str] = None
    doc_type: Optional[str] = None
    metadata: dict = {}


class DocumentUploadResponse(BaseModel):
    id: str
    title: str
    content: str
    doc_type: str = "default"
    status: str = "completed"
    metadata: dict = {}
    created_at: str
    updated_at: str
    performance: Optional[dict] = None


@router.get("", response_model=List[DocumentResponse])
async def get_documents(db: Session = Depends(get_session), storage=Depends(get_storage_service)):
    """Get all documents."""
    documents = await storage.get_documents()
    
    # If no documents in storage, return empty list
    if not documents:
        return []
    
    # Convert to response format
    return [
        DocumentResponse(
            id=doc.get("id"),
            title=doc.get("title") or "Untitled",  # Handle None title
            content=doc.get("content"),
            created_at=doc.get("created_at"),
            doc_type=doc.get("doc_type", "default"),
            metadata=doc.get("metadata", {})
        )
        for doc in documents
    ]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, db: Session = Depends(get_session), storage=Depends(get_storage_service)):
    """Get a specific document by ID."""
    document = await storage.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        id=document.get("id"),
        title=document.get("title"),
        content=document.get("content"),
        created_at=document.get("created_at"),
        doc_type=document.get("doc_type", "default"),
        metadata=document.get("metadata", {})
    )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_session),
    storage=Depends(get_storage_service)
):
    """Upload and process a document with performance tracking."""
    
    # Initialize performance tracking
    perf_times = {}
    total_start = time.time()
    
    # Read file content
    read_start = time.time()
    content = await file.read()
    perf_times['file_read'] = round(time.time() - read_start, 3)
    
    # Determine file type
    file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
    
    # Process based on file type
    process_start = time.time()
    if file_extension in ['pdf'] and TEXTRACT_AVAILABLE:
        # Use Textract for PDF processing
        try:
            textract_processor = TextractProcessor()
            text_content = textract_processor.process_document(content, file.filename)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    elif file_extension in ['png', 'jpg', 'jpeg', 'tiff', 'bmp'] and TEXTRACT_AVAILABLE:
        # Use Textract for image processing
        try:
            textract_processor = TextractProcessor()
            text_content = textract_processor.process_document(content, file.filename)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
    else:
        # Try to decode as text
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            if file_extension in ['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp']:
                raise HTTPException(status_code=400, detail="Textract is not available. Cannot process PDF/image files.")
            else:
                raise HTTPException(status_code=400, detail="Unable to process this file type. Only text, PDF, and image files are supported.")
    perf_times['content_processing'] = round(time.time() - process_start, 3)
    
    # Generate document ID
    doc_id = hashlib.md5(content).hexdigest()[:12]
    
    # Extract temporal metadata
    temporal_metadata = extract_temporal_metadata(text_content, file.filename)
    
    # Enrich with temporal information
    metadata = enrich_with_temporal_metadata(
        {},
        text_content,
        file.filename
    )
    
    # Set document type based on file extension
    if file_extension in ['pdf']:
        metadata['doc_type'] = 'pdf'
    elif file_extension in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
        metadata['doc_type'] = 'image'
    elif file_extension in ['md', 'markdown']:
        metadata['doc_type'] = 'markdown'
    else:
        metadata['doc_type'] = 'text'
    
    # Create chunks using hierarchical chunker for better structure
    chunk_start = time.time()
    hierarchical_chunker = HierarchicalChunker(chunk_size=400, chunk_overlap=80)
    
    # Try hierarchical chunking first
    hierarchical_chunks = hierarchical_chunker.chunk_hierarchical(text_content, doc_id)
    
    chunks = []
    if hierarchical_chunks:
        # Convert hierarchical chunks to storage format
        for h_chunk in hierarchical_chunks:
            chunk = {
                "id": h_chunk.id,
                "document_id": doc_id,
                "content": h_chunk.content,
                "position": h_chunk.chunk_index,
                "chunk_index": h_chunk.chunk_index,
                "tokens": h_chunk.tokens,
                "chunk_type": "hierarchical",
                "parent_id": h_chunk.parent_id,
                "children_ids": h_chunk.children_ids,
                "level": h_chunk.level,
                "metadata": {
                    **metadata,
                    **h_chunk.metadata,
                    "chunking_strategy": "hierarchical",
                    "chunk_size": 400,
                    "chunk_overlap": 80
                }
            }
            chunks.append(chunk)
    else:
        # Fallback to semantic chunking
        chunker = SemanticChunker(chunk_size=400, chunk_overlap=80)
        text_chunks = chunker.chunk(text_content)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                "id": f"{doc_id}_chunk_{i}",
                "document_id": doc_id,
                "content": chunk_text,
                "position": i,
                "chunk_index": i,
                "tokens": chunker.count_tokens(chunk_text),
                "chunk_type": "semantic",
                "parent_id": None,
                "children_ids": [],
                "level": 0,
                "metadata": {
                    **metadata,
                    "chunking_strategy": "semantic",
                    "chunk_size": 400,
                    "chunk_overlap": 80
                }
            }
            chunks.append(chunk)
    
    perf_times['chunking'] = round(time.time() - chunk_start, 3)
    perf_times['chunk_count'] = len(chunks)
    perf_times['hierarchical'] = len(hierarchical_chunks) > 0
    
    # Generate embeddings and store vectors
    embedding_start = time.time()
    try:
        if chunks:
            logger.info(f"🔍 Generating embeddings for {len(chunks)} chunks...")
            embedding_service = get_embedding_service()
            vector_service = get_vector_service()
            
            # Extract text content from chunks for embedding
            chunk_texts = [chunk["content"] for chunk in chunks]
            embeddings = await embedding_service.generate_embeddings(chunk_texts)
            
            if embeddings:
                logger.info(f"📦 Storing {len(embeddings)} vectors in Qdrant...")
                vector_result = await vector_service.store_vectors(chunks, embeddings)
                if vector_result:
                    logger.info(f"✅ Successfully stored vectors for document {doc_id}")
                    metadata["vectors_stored"] = True
                    metadata["vector_count"] = len(embeddings)
                else:
                    logger.warning(f"⚠️ Failed to store vectors for document {doc_id}")
                    metadata["vectors_stored"] = False
            else:
                logger.warning(f"⚠️ No embeddings generated for document {doc_id}")
                metadata["vectors_stored"] = False
        perf_times['embedding_generation'] = round(time.time() - embedding_start, 3)
    except Exception as e:
        logger.error(f"❌ Embedding/vector storage failed: {e}")
        metadata["vectors_stored"] = False
        perf_times['embedding_generation'] = round(time.time() - embedding_start, 3)
    
    # Extract entities and relationships (graph extraction)
    extract_start = time.time()
    entity_count = 0
    relationship_count = 0
    try:
        entity_service = get_entity_service()
        if entity_service and entity_service.initialized:
            chunk_ids = [chunk["id"] for chunk in chunks]
            entities, relationships = await entity_service.extract_entities(
                text=text_content,
                document_id=doc_id,
                chunk_ids=chunk_ids,
                use_claude=True  # Use Claude for better extraction
            )
            entity_count = len(entities) if entities else 0
            relationship_count = len(relationships) if relationships else 0
            
            # Store entities and relationships if any were extracted
            storage_start = time.time()
            if entities:
                print(f"📊 Storing {len(entities)} entities...")
                # Store entities in both Supabase and Neo4j
                supabase_result = await storage.store_entities(entities)
                print(f"  Supabase storage: {'✅' if supabase_result else '❌'}")
                
                # Also try to store in Neo4j if available
                try:
                    from ..services.graph_service import GraphService
                    graph_service = GraphService()
                    print(f"  Neo4j initialized: {graph_service.initialized}")
                    if graph_service.initialized:
                        neo4j_result = await graph_service.store_entities(entities)
                        print(f"  Neo4j storage: {'✅' if neo4j_result else '❌'}")
                    else:
                        print(f"  Neo4j not initialized, skipping")
                except Exception as e:
                    print(f"  Neo4j storage failed (continuing): {e}")
                
                metadata["entities_count"] = len(entities)
            
            if relationships:
                # Store relationships in both Supabase and Neo4j
                await storage.store_relationships(relationships)
                
                # Also try to store in Neo4j if available
                try:
                    from ..services.graph_service import GraphService
                    graph_service = GraphService()
                    if graph_service.initialized:
                        await graph_service.store_relationships(relationships)
                except Exception as e:
                    print(f"Neo4j relationship storage failed (continuing): {e}")
                
                metadata["relationships_count"] = len(relationships)
            perf_times['entity_storage'] = round(time.time() - storage_start, 3)
    except Exception as e:
        print(f"⚠️ Entity extraction failed: {e}")
        # Continue even if entity extraction fails
    perf_times['entity_extraction'] = round(time.time() - extract_start, 3)
    perf_times['entity_count'] = entity_count
    perf_times['relationship_count'] = relationship_count
    
    # Store document in Supabase
    storage_start = time.time()
    now = datetime.now().isoformat()
    
    document = {
        "id": doc_id,
        "title": file.filename,
        "content": text_content,
        "doc_type": metadata.get('doc_type', 'default'),
        "status": "completed",
        "metadata": metadata,
        "created_at": now,
        "updated_at": now
    }
    
    stored_doc = await storage.store_document(document)
    
    # Store chunks in Supabase
    if chunks:
        await storage.store_chunks(chunks)
    perf_times['document_storage'] = round(time.time() - storage_start, 3)
    
    # Calculate total time
    perf_times['total_time'] = round(time.time() - total_start, 3)
    perf_times['content_length'] = len(text_content)
    
    return DocumentUploadResponse(
        id=doc_id,
        title=file.filename,
        content=text_content,
        doc_type=metadata.get('doc_type', 'default'),
        status="completed",
        metadata=metadata,
        created_at=now,
        updated_at=now,
        performance=perf_times
    )


@router.delete("/{document_id}")
async def delete_document(document_id: str, db: Session = Depends(get_session), storage=Depends(get_storage_service)):
    """Delete a document and its chunks."""
    success = await storage.delete_document(document_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": f"Document {document_id} deleted successfully"}


@router.get("/{document_id}/chunks", response_model=List[Dict])
async def get_document_chunks(document_id: str, db: Session = Depends(get_session), storage=Depends(get_storage_service)):
    """Get all chunks for a document."""
    chunks = await storage.get_chunks(document_id)
    
    # Return array directly for frontend compatibility
    return chunks if chunks else []