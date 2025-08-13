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
        self.client = None
        self.initialized = False
        
        supabase_url = os.getenv("SUPABASE_URL", "")
        supabase_key = os.getenv("SUPABASE_ANON_API_KEY", "")
        
        if supabase_url and supabase_key:
            try:
                self.client: Client = create_client(supabase_url, supabase_key)
                self.initialized = True
                logger.info("Supabase client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Supabase client: {e}. Storage will not persist.")
                self.client = None
                self.initialized = False
        else:
            logger.info("Supabase credentials not configured, storage will not persist")
    
    async def store_document(self, document: Dict) -> Dict:
        """Store a document in Supabase"""
        if not self.initialized:
            logger.warning("Supabase not initialized, returning document without persisting")
            return document
        
        try:
            # Check if document already exists
            existing = self.client.table("rag_documents").select("id").eq("id", document["id"]).execute()
            
            if existing.data:
                # Update existing document instead of creating duplicate
                result = self.client.table("rag_documents").update({
                    "title": document["title"],
                    "content": document.get("content", ""),
                    "doc_type": document.get("doc_type", "default"),
                    "status": document.get("status", "completed"),
                    "metadata": document.get("metadata", {}),
                    "updated_at": document.get("updated_at", datetime.now().isoformat())
                }).eq("id", document["id"]).execute()
                logger.info(f"Updated existing document {document['id']}")
            else:
                # Insert new document
                result = self.client.table("rag_documents").insert({
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
            result = self.client.table("rag_documents").select("*").execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Error fetching documents: {e}")
            return []
    
    async def get_document(self, document_id: str) -> Optional[Dict]:
        """Get a specific document from Supabase"""
        if not self.initialized:
            return None
        
        try:
            result = self.client.table("rag_documents").select("*").eq("id", document_id).execute()
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
            self.client.table("rag_chunks").delete().eq("document_id", document_id).execute()
            # Delete document
            self.client.table("rag_documents").delete().eq("id", document_id).execute()
            logger.info(f"Document {document_id} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    async def store_entities(self, entities: List[Dict]) -> bool:
        """Store entities in Supabase"""
        if not self.initialized or not entities:
            return False
        
        try:
            # Batch insert entities
            entity_records = []
            for entity in entities:
                entity_records.append({
                    "id": entity["id"],
                    "name": entity["name"],
                    "entity_type": entity.get("entity_type", entity.get("type", "Unknown")),
                    "document_ids": entity.get("document_ids", []),
                    "chunk_ids": entity.get("chunk_ids", []),
                    "frequency": entity.get("frequency", 1),
                    "metadata": entity.get("metadata", {}),
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                })
            
            # Upsert entities (update if exists, insert if not)
            result = self.client.table("entities").upsert(entity_records).execute()
            logger.info(f"Stored {len(entities)} entities in Supabase")
            return True
        except Exception as e:
            logger.error(f"Error storing entities in Supabase: {e}")
            return False
    
    async def store_relationships(self, relationships: List[Dict]) -> bool:
        """Store relationships in Supabase"""
        if not self.initialized or not relationships:
            return False
        
        try:
            # Batch insert relationships
            relationship_records = []
            for rel in relationships:
                relationship_records.append({
                    "id": rel.get("id", f"{rel.get('source_entity_id')}_{rel.get('target_entity_id')}"),
                    "source_entity_id": rel.get("source_entity_id", rel.get("source_id")),
                    "target_entity_id": rel.get("target_entity_id", rel.get("target_id")),
                    "relationship_type": rel.get("relationship_type", rel.get("type", "related_to")),
                    "weight": rel.get("weight", 1.0),
                    "document_ids": rel.get("document_ids", []),
                    "metadata": rel.get("metadata", {}),
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                })
            
            # Upsert relationships
            result = self.client.table("relationships").upsert(relationship_records).execute()
            logger.info(f"Stored {len(relationships)} relationships in Supabase")
            return True
        except Exception as e:
            logger.error(f"Error storing relationships in Supabase: {e}")
            return False
    
    async def get_entities(self, document_id: Optional[str] = None) -> List[Dict]:
        """Get entities from Supabase"""
        if not self.initialized:
            return []
        
        try:
            query = self.client.table("entities").select("*")
            if document_id:
                # Filter by document_id in the document_ids array
                query = query.contains("document_ids", [document_id])
            
            result = query.execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Error fetching entities: {e}")
            return []
    
    async def get_relationships(self, document_id: Optional[str] = None) -> List[Dict]:
        """Get relationships from Supabase"""
        if not self.initialized:
            return []
        
        try:
            query = self.client.table("relationships").select("*")
            if document_id:
                # Filter by document_id in the document_ids array
                query = query.contains("document_ids", [document_id])
            
            result = query.execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Error fetching relationships: {e}")
            return []
    
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
            
            result = self.client.table("rag_chunks").insert(chunk_records).execute()
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
            result = self.client.table("rag_chunks").select("*").eq("document_id", document_id).order("chunk_index").execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Error fetching chunks for document {document_id}: {e}")
            return []
    
    async def clear_all(self) -> Dict:
        """Clear all data from Supabase tables"""
        if not self.initialized:
            return {"status": "error", "message": "Supabase not initialized"}
        
        try:
            # First, get counts before deletion for reporting
            docs_count = len(self.client.table("rag_documents").select("id").execute().data or [])
            chunks_count = len(self.client.table("rag_chunks").select("id").execute().data or [])
            entities_count = len(self.client.table("entities").select("id").execute().data or [])
            relationships_count = len(self.client.table("relationships").select("id").execute().data or [])
            
            # Delete all relationships first (due to foreign key constraints)
            # Use gte with 0 to select all records (works for any id format)
            relationships_result = self.client.table("relationships").delete().gte("created_at", "1970-01-01").execute()
            
            # Delete all entities
            entities_result = self.client.table("entities").delete().gte("created_at", "1970-01-01").execute()
            
            # Delete all chunks
            chunks_result = self.client.table("rag_chunks").delete().gte("created_at", "1970-01-01").execute()
            
            # Delete all documents
            docs_result = self.client.table("rag_documents").delete().gte("created_at", "1970-01-01").execute()
            
            logger.info(f"Cleared {docs_count} documents, {chunks_count} chunks, {entities_count} entities, and {relationships_count} relationships")
            
            return {
                "status": "success",
                "documents_deleted": docs_count,
                "chunks_deleted": chunks_count,
                "entities_deleted": entities_count,
                "relationships_deleted": relationships_count
            }
        except Exception as e:
            logger.error(f"Error clearing all data: {e}")
            return {"status": "error", "message": str(e)}


# Dependency injection for lazy initialization
def get_storage_service():
    """Get or create storage service instance"""
    if not hasattr(get_storage_service, "_instance"):
        get_storage_service._instance = StorageService()
    return get_storage_service._instance