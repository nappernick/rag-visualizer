"""
Entity extraction service using SpaCy
"""
import os
from typing import List, Dict, Optional, Tuple
import logging
import spacy
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)


class EntityService:
    """Service for extracting entities from text using SpaCy"""
    
    def __init__(self):
        model_name = os.getenv("SPACY_MODEL", "en_core_web_sm")
        
        try:
            self.nlp = spacy.load(model_name)
            self.initialized = True
            logger.info(f"SpaCy model loaded: {model_name}")
        except OSError:
            logger.warning(f"SpaCy model '{model_name}' not found, using fallback extraction")
            self.nlp = None
            self.initialized = False
    
    async def extract_entities(
        self, 
        text: str, 
        document_id: str,
        chunk_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """Extract entities from text"""
        
        if self.initialized and self.nlp:
            return await self._extract_with_spacy(text, document_id, chunk_ids)
        else:
            return await self._extract_fallback(text, document_id, chunk_ids)
    
    async def _extract_with_spacy(
        self, 
        text: str, 
        document_id: str,
        chunk_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """Extract entities using SpaCy NLP"""
        
        try:
            doc = self.nlp(text)
            entity_counts = Counter()
            entities = []
            
            # Extract named entities
            for ent in doc.ents:
                if len(ent.text.strip()) < 2:  # Skip very short entities
                    continue
                    
                entity_text = ent.text.strip()
                entity_type = ent.label_
                
                # Skip common but not useful entities
                if entity_type in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                    continue
                
                entity_counts[entity_text] += 1
                
                # Create entity ID based on text
                entity_id = hashlib.md5(f"{entity_text}_{entity_type}".encode()).hexdigest()[:12]
                
                entity = {
                    "id": entity_id,
                    "name": entity_text,
                    "entity_type": entity_type,
                    "document_ids": [document_id],
                    "chunk_ids": chunk_ids or [],
                    "frequency": 1,  # Will be updated later
                    "metadata": {
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "confidence": 1.0
                    }
                }
                
                entities.append(entity)
            
            # Update frequencies
            entity_freq_map = {}
            for entity in entities:
                key = entity["name"]
                if key in entity_freq_map:
                    entity_freq_map[key]["frequency"] += 1
                else:
                    entity_freq_map[key] = entity
            
            # Return unique entities with correct frequencies
            unique_entities = list(entity_freq_map.values())
            
            logger.info(f"Extracted {len(unique_entities)} unique entities using SpaCy")
            return unique_entities
            
        except Exception as e:
            logger.error(f"Error extracting entities with SpaCy: {e}")
            return await self._extract_fallback(text, document_id, chunk_ids)
    
    async def _extract_fallback(
        self, 
        text: str, 
        document_id: str,
        chunk_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """Fallback entity extraction using simple patterns"""
        
        entities = []
        text_lower = text.lower()
        
        # Simple keyword-based extraction
        tech_keywords = {
            "redis": "Technology",
            "cache": "Concept",
            "memory": "Concept", 
            "database": "Technology",
            "graph": "Concept",
            "vector": "Concept",
            "embedding": "Concept",
            "search": "Concept",
            "query": "Concept",
            "document": "Concept",
            "chunk": "Concept",
            "entity": "Concept",
            "relationship": "Concept",
            "python": "Technology",
            "fastapi": "Technology",
            "supabase": "Technology",
            "qdrant": "Technology",
            "neo4j": "Technology"
        }
        
        for keyword, entity_type in tech_keywords.items():
            if keyword in text_lower:
                entity_id = hashlib.md5(f"{keyword}_{entity_type}".encode()).hexdigest()[:12]
                
                entity = {
                    "id": entity_id,
                    "name": keyword.title(),
                    "entity_type": entity_type,
                    "document_ids": [document_id],
                    "chunk_ids": chunk_ids or [],
                    "frequency": text_lower.count(keyword),
                    "metadata": {
                        "extraction_method": "fallback",
                        "confidence": 0.7
                    }
                }
                
                entities.append(entity)
        
        logger.info(f"Extracted {len(entities)} entities using fallback method")
        return entities
    
    async def extract_relationships(
        self, 
        entities: List[Dict],
        text: str,
        document_id: str
    ) -> List[Dict]:
        """Extract relationships between entities"""
        
        relationships = []
        
        if len(entities) < 2:
            return relationships
        
        # Simple co-occurrence based relationships
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i + 1:], i + 1):
                
                # Create relationship if entities co-occur
                rel_id = f"{entity1['id']}_{entity2['id']}"
                
                # Determine relationship type based on entity types
                rel_type = self._determine_relationship_type(
                    entity1["entity_type"], 
                    entity2["entity_type"]
                )
                
                # Calculate weight based on frequency and proximity
                weight = min(entity1["frequency"], entity2["frequency"]) / max(entity1["frequency"], entity2["frequency"])
                weight = max(0.1, weight)  # Minimum weight
                
                relationship = {
                    "id": rel_id,
                    "source_entity_id": entity1["id"],
                    "target_entity_id": entity2["id"],
                    "relationship_type": rel_type,
                    "weight": weight,
                    "document_ids": [document_id],
                    "metadata": {
                        "extraction_method": "co_occurrence",
                        "confidence": weight
                    }
                }
                
                relationships.append(relationship)
        
        logger.info(f"Extracted {len(relationships)} relationships")
        return relationships
    
    def _determine_relationship_type(self, type1: str, type2: str) -> str:
        """Determine relationship type based on entity types"""
        
        if type1 == type2:
            return "similar_to"
        
        if (type1 == "Technology" and type2 == "Concept") or (type1 == "Concept" and type2 == "Technology"):
            return "implements"
        
        if type1 == "PERSON" and type2 == "ORG":
            return "works_for"
        
        if type1 == "ORG" and type2 == "GPE":
            return "located_in"
        
        return "related_to"


# Global entity service instance
entity_service = EntityService()