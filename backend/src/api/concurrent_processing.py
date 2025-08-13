"""
Concurrent processing API for document upload, chunking, and graph extraction
"""
import asyncio
import hashlib
from typing import Dict, List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
import logging

from ..db import get_session
from ..core.temporal.temporal_utils import enrich_with_temporal_metadata
from ..core.temporal.date_extractor import extract_temporal_metadata
from ..core.chunking.semantic_chunker import SemanticChunker
from ..core.chunking.hierarchical import HierarchicalChunker
from ..core.chunking.base import StandardChunker
from ..core.graph.claude_extractor import ClaudeGraphExtractor
from ..models import Document as DocumentSchema
from ..services.storage import get_storage_service
from ..services.vector_service import get_vector_service
from ..services.embedding_service import get_embedding_service
from ..services.entity_service import get_entity_service
from ..services.graph_service import get_graph_service

# Try to import Textract processor
try:
    from ..core.ocr.textract_processor import TextractProcessor
    TEXTRACT_AVAILABLE = True
except ImportError:
    TextractProcessor = None
    TEXTRACT_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["concurrent_processing"])


class ProcessingStatus(BaseModel):
    stage: str  # "uploading", "extracting_text", "chunking", "extracting_graph", "storing", "completed"
    progress: int  # 0-100
    message: str
    details: Optional[Dict] = None


class ProcessingResponse(BaseModel):
    document_id: str
    title: str
    status: ProcessingStatus
    chunks_created: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    processing_time_ms: float = 0


async def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract text from uploaded file using appropriate method"""
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
    
    # Process based on file type
    if file_extension in ['pdf'] and TEXTRACT_AVAILABLE:
        # Use Textract for PDF processing
        textract_processor = TextractProcessor()
        return textract_processor.process_document(file_content, filename)
    elif file_extension in ['png', 'jpg', 'jpeg', 'tiff', 'bmp'] and TEXTRACT_AVAILABLE:
        # Use Textract for image processing
        textract_processor = TextractProcessor()
        return textract_processor.process_document(file_content, filename)
    else:
        # Try to decode as text
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            if file_extension in ['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp']:
                raise HTTPException(status_code=400, detail="Textract is not available. Cannot process PDF/image files.")
            else:
                raise HTTPException(status_code=400, detail="Unable to process this file type. Only text, PDF, and image files are supported.")


async def chunk_document(document_id: str, content: str, strategy: str = "semantic", 
                        max_chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
    """Chunk document content"""
    chunks = []
    
    # Create a mock document object for the chunkers
    mock_document = DocumentSchema(
        id=document_id,
        title="Document",
        content=content
    )
    
    # Choose chunker based on strategy
    if strategy == "hierarchical":
        chunker = HierarchicalChunker(
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            create_summaries=True
        )
        chunk_objects = chunker.chunk_document(mock_document)
    elif strategy == "semantic":
        try:
            chunker = SemanticChunker(
                chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunk_texts = chunker.chunk(content)
            # Convert to Chunk objects
            chunk_objects = []
            for i, text in enumerate(chunk_texts):
                chunk_id = hashlib.md5(f"{document_id}_{i}_{text[:50]}".encode()).hexdigest()[:12]
                chunk_obj = type('Chunk', (), {
                    'id': chunk_id,
                    'content': text,
                    'document_id': document_id,
                    'chunk_index': i,
                    'chunk_type': 'standard',
                    'tokens': len(text.split()),
                    'metadata': {'method': 'semantic'},
                    'parent_id': None,
                    'children_ids': [],
                    'created_at': datetime.now().isoformat()
                })()
                chunk_objects.append(chunk_obj)
        except Exception:
            # Fall back to standard chunking
            chunker = StandardChunker(
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunk_objects = chunker.chunk_document(mock_document)
    else:
        # Standard chunking
        chunker = StandardChunker(
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunk_objects = chunker.chunk_document(mock_document)
    
    # Convert chunk objects to dictionaries
    for chunk_obj in chunk_objects:
        chunk_id = getattr(chunk_obj, 'id', hashlib.md5(f"{document_id}_{len(chunks)}".encode()).hexdigest()[:12])
        chunk_content = getattr(chunk_obj, 'content', '')
        chunk_index = getattr(chunk_obj, 'chunk_index', len(chunks))
        chunk_type = getattr(chunk_obj, 'chunk_type', 'standard')
        tokens = getattr(chunk_obj, 'tokens', len(chunk_content.split()))
        metadata = getattr(chunk_obj, 'metadata', {})
        parent_id = getattr(chunk_obj, 'parent_id', None)
        children_ids = getattr(chunk_obj, 'children_ids', [])
        created_at_raw = getattr(chunk_obj, 'created_at', datetime.now())
        created_at = created_at_raw.isoformat() if isinstance(created_at_raw, datetime) else str(created_at_raw)
        
        metadata.update({
            "method": strategy,
            "chunk_size": max_chunk_size,
            "overlap": chunk_overlap
        })
        
        chunks.append({
            "id": chunk_id,
            "document_id": document_id,
            "content": chunk_content,
            "chunk_index": chunk_index,
            "chunk_type": chunk_type,
            "tokens": tokens,
            "metadata": metadata,
            "parent_id": parent_id,
            "children_ids": children_ids,
            "created_at": created_at
        })
    
    return chunks


async def extract_graph(content: bytes, document_id: str, filename: str) -> tuple[List[Dict], List[Dict]]:
    """Extract entities and relationships from document"""
    entity_service = get_entity_service()
    
    # Extract entities and relationships using Claude
    result = await entity_service.extract_entities_from_image(
        image_data=content,
        document_id=document_id,
        chunk_id=f"{document_id}_full"
    )
    
    entities = result.get("entities", [])
    relationships = result.get("relationships", [])
    
    return entities, relationships


@router.post("/process", response_model=ProcessingResponse)
async def process_document_concurrent(
    file: UploadFile = File(...),
    chunking_strategy: str = "semantic",
    max_chunk_size: int = 500,
    chunk_overlap: int = 50,
    db: Session = Depends(get_session)
):
    """
    Process a document with concurrent chunking and graph extraction.
    Returns immediately with status updates.
    """
    start_time = datetime.now()
    
    # Read file content
    content = await file.read()
    
    # Generate document ID
    doc_id = hashlib.md5(content).hexdigest()[:12]
    
    # Initialize services
    storage_service = get_storage_service()
    vector_service = get_vector_service()
    embedding_service = get_embedding_service()
    graph_service = get_graph_service()
    
    try:
        # Step 1: Extract text from file
        logger.info(f"Extracting text from {file.filename}")
        text_content = await extract_text_from_file(content, file.filename)
        
        # Step 2: Create document metadata
        file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
        metadata = enrich_with_temporal_metadata({}, text_content, file.filename)
        
        if file_extension in ['pdf']:
            metadata['doc_type'] = 'pdf'
        elif file_extension in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
            metadata['doc_type'] = 'image'
        elif file_extension in ['md', 'markdown']:
            metadata['doc_type'] = 'markdown'
        else:
            metadata['doc_type'] = 'text'
        
        # Step 3: Store document
        now = datetime.now().isoformat()
        document = {
            "id": doc_id,
            "title": file.filename,
            "content": text_content,
            "doc_type": metadata.get('doc_type', 'default'),
            "status": "processing",
            "metadata": metadata,
            "created_at": now,
            "updated_at": now
        }
        await storage_service.store_document(document)
        
        # Step 4: Run chunking and graph extraction concurrently
        logger.info(f"Starting concurrent processing for document {doc_id}")
        
        # Create concurrent tasks
        chunking_task = asyncio.create_task(
            chunk_document(doc_id, text_content, chunking_strategy, max_chunk_size, chunk_overlap)
        )
        
        graph_task = asyncio.create_task(
            extract_graph(content, doc_id, file.filename)
        )
        
        # Wait for both tasks to complete
        chunks, (entities, relationships) = await asyncio.gather(
            chunking_task,
            graph_task
        )
        
        logger.info(f"Concurrent processing completed: {len(chunks)} chunks, {len(entities)} entities, {len(relationships)} relationships")
        
        # Step 5: Store chunks and generate embeddings
        if chunks:
            await storage_service.store_chunks(chunks)
            
            # Generate embeddings for chunks
            chunk_texts = [chunk["content"] for chunk in chunks]
            embeddings = await embedding_service.generate_embeddings(chunk_texts)
            
            if embeddings:
                await vector_service.store_vectors(chunks, embeddings)
        
        # Step 6: Store entities and relationships
        if entities:
            await storage_service.store_entities(entities)
            await graph_service.store_entities(entities)
        
        if relationships:
            await storage_service.store_relationships(relationships)
            await graph_service.store_relationships(relationships)
        
        # Step 7: Update document status
        document["status"] = "completed"
        await storage_service.store_document(document)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ProcessingResponse(
            document_id=doc_id,
            title=file.filename,
            status=ProcessingStatus(
                stage="completed",
                progress=100,
                message="Document processed successfully",
                details={
                    "chunks": len(chunks),
                    "entities": len(entities),
                    "relationships": len(relationships)
                }
            ),
            chunks_created=len(chunks),
            entities_extracted=len(entities),
            relationships_extracted=len(relationships),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        
        # Update document status to failed
        try:
            document["status"] = "failed"
            document["metadata"]["error"] = str(e)
            await storage_service.store_document(document)
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")