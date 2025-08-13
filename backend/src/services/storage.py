"""
Storage service for managing documents, chunks, and metadata
Uses Supabase for persistent storage
"""
import os
from typing import List, Dict, Optional, Any
from datetime import datetime
import hashlib
from supabase import create_client, Client
import logging

logger = logging.getLogger(__name__)


class StorageService:
    """Service for managing document and chunk storage in Supabase"""
    
    def __init__(self):
        supabase_url = os.getenv("SUPABASE_URL", "")
        supabase_key = os.getenv("SUPABASE_ANON_API_KEY", "")
        
        if supabase_url and supabase_key:
            self.client: Client = create_client(supabase_url, supabase_key)
            self.initialized = True
            logger.info("Supabase client initialized successfully")
        else:
            self.client = None
            self.initialized = False
            logger.warning("Supabase credentials not found, storage will not persist")
    
    async def store_document(self, document: Dict) -> Dict:
        """Store a document in Supabase"""
        if not self.initialized:
            logger.warning("Supabase not initialized, returning document without persisting")
            return document
        
        try:
            # Insert into documents table
            result = self.client.table("documents").insert({
                "id": document["id"],
                "title": document["title"],
                "content": document.get("content", ""),
                "doc_type": document.get("doc_type", "default"),
                "status": document.get("status", "completed"),
                "metadata": document.get("metadata", {}),
                "created_at": document.get("created_at", datetime.now().isoformat()),
                "updated_at": document.get("updated_at", datetime.now().isoformat())
            }).execute()
            
            logger.info(f"Document {document['id']} stored successfully")
            return result.data[0] if result.data else document
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            return document
    
    async def get_documents(self) -> List[Dict]:
        """Get all documents from Supabase"""
        if not self.initialized:
            logger.warning("Supabase not initialized, returning empty list")
            return []
        
        try:
            result = self.client.table("documents").select("*").execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Error fetching documents: {e}")
            return []
    
    async def get_document(self, document_id: str) -> Optional[Dict]:
        """Get a specific document from Supabase"""
        if not self.initialized:
            return None
        
        try:
            result = self.client.table("documents").select("*").eq("id", document_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error fetching document {document_id}: {e}")
            return None
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks from Supabase"""
        if not self.initialized:
            return False
        
        try:
            # Delete chunks first
            self.client.table("chunks").delete().eq("document_id", document_id).execute()
            # Delete document
            self.client.table("documents").delete().eq("id", document_id).execute()
            logger.info(f"Document {document_id} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    async def store_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Store multiple chunks in Supabase"""
        if not self.initialized or not chunks:
            return chunks
        
        try:
            # Prepare chunks for insertion
            chunk_records = []
            for chunk in chunks:
                chunk_records.append({
                    "id": chunk["id"],
                    "document_id": chunk["document_id"],
                    "content": chunk["content"],
                    "chunk_index": chunk.get("chunk_index", 0),
                    "chunk_type": chunk.get("chunk_type", "standard"),
                    "tokens": chunk.get("tokens", 0),
                    "metadata": chunk.get("metadata", {}),
                    "parent_id": chunk.get("parent_id"),
                    "children_ids": chunk.get("children_ids", []),
                    "created_at": chunk.get("created_at", datetime.now().isoformat())
                })
            
            result = self.client.table("chunks").insert(chunk_records).execute()
            logger.info(f"Stored {len(chunk_records)} chunks successfully")
            return result.data if result.data else chunks
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            return chunks
    
    async def get_chunks(self, document_id: str) -> List[Dict]:
        """Get all chunks for a document from Supabase"""
        if not self.initialized:
            return []
        
        try:
            result = self.client.table("chunks").select("*").eq("document_id", document_id).order("chunk_index").execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Error fetching chunks for document {document_id}: {e}")
            return []
    
    async def clear_all(self) -> Dict:
        """Clear all data from Supabase tables"""
        if not self.initialized:
            return {"status": "error", "message": "Supabase not initialized"}
        
        try:
            # Delete all chunks
            chunks_result = self.client.table("chunks").delete().neq("id", "").execute()
            chunks_deleted = len(chunks_result.data) if chunks_result.data else 0
            
            # Delete all documents
            docs_result = self.client.table("documents").delete().neq("id", "").execute()
            docs_deleted = len(docs_result.data) if docs_result.data else 0
            
            logger.info(f"Cleared {docs_deleted} documents and {chunks_deleted} chunks")
            
            return {
                "status": "success",
                "documents_deleted": docs_deleted,
                "chunks_deleted": chunks_deleted
            }
        except Exception as e:
            logger.error(f"Error clearing all data: {e}")
            return {"status": "error", "message": str(e)}


# Global storage service instance
storage_service = StorageService()