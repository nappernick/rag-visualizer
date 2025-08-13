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

from ..db import get_session
from ..models import Document, Chunk
from ..core.temporal.temporal_utils import enrich_with_temporal_metadata
from ..core.temporal.date_extractor import extract_temporal_metadata
from ..services.storage import get_storage_service

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
    """Upload and process a document."""
    
    # Read file content
    content = await file.read()
    
    # Determine file type
    file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
    
    # Process based on file type
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
    
    # Create chunks (simple splitting for now)
    chunks = []
    chunk_size = 500
    text_chunks = [text_content[i:i+chunk_size] 
                   for i in range(0, len(text_content), chunk_size)]
    
    for i, chunk_text in enumerate(text_chunks):
        chunk = {
            "id": f"{doc_id}_chunk_{i}",
            "document_id": doc_id,
            "content": chunk_text,
            "position": i,
            "metadata": metadata
        }
        chunks.append(chunk)
    
    # Store document in Supabase
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
    
    return DocumentUploadResponse(
        id=doc_id,
        title=file.filename,
        content=text_content,
        doc_type=metadata.get('doc_type', 'default'),
        status="completed",
        metadata=metadata,
        created_at=now,
        updated_at=now
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