"""
Graph Builder Service - Handles graph construction and maintenance
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
from datetime import datetime
import hashlib
import json

from ...models.schemas import Document, Chunk, Entity, Relationship
from .entity_service import EntityService
from .relationship_service import RelationshipService

logger = logging.getLogger(__name__)


class GraphBuilderService:
    """Service for building and maintaining the knowledge graph"""
    
    def __init__(self, graph_store=None, nlp_processor=None):
        self.graph_store = graph_store
        self.nlp_processor = nlp_processor
        self.entity_service = EntityService(graph_store)
        self.relationship_service = RelationshipService(graph_store)
        
        # Entity extraction patterns
        self.entity_patterns = {
            'person': r'\b(?:[A-Z][a-z]+ ){1,3}[A-Z][a-z]+\b',
            'organization': r'\b(?:[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*(?:\s+(?:Inc|Corp|LLC|Ltd|Company|Group|Technologies|Systems|Solutions))?)\b',
            'location': r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s*[A-Z][a-z]+)?)\b',
            'date': r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'https?://[^\s]+',
            'phone': r'\b(?:\+?1?[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        }
    
    async def build_graph_from_document(
        self,
        document: Document,
        chunks: List[Chunk],
        extract_entities: bool = True,
        extract_relationships: bool = True,
        merge_similar: bool = True
    ) -> Dict[str, Any]:
        """
        Build a knowledge graph from a document and its chunks
        
        Args:
            document: The document to process
            chunks: Document chunks
            extract_entities: Whether to extract entities
            extract_relationships: Whether to extract relationships
            merge_similar: Whether to merge similar entities
            
        Returns:
            Graph building results
        """
        entities = []
        relationships = []
        
        # Extract entities from each chunk
        if extract_entities:
            for chunk in chunks:
                chunk_entities = await self._extract_entities_from_text(
                    chunk.content,
                    document.id,
                    chunk.id
                )
                entities.extend(chunk_entities)
        
        # Merge similar entities if requested
        if merge_similar and entities:
            entities = await self._merge_similar_entities(entities)
        
        # Extract relationships
        if extract_relationships and entities:
            for chunk in chunks:
                chunk_relationships = await self._extract_relationships_from_text(
                    chunk.content,
                    entities,
                    document.id,
                    chunk.id
                )
                relationships.extend(chunk_relationships)
        
        # Store in graph
        stored_entities = []
        stored_relationships = []
        
        for entity in entities:
            stored_entity = await self.entity_service.create_entity(
                name=entity['name'],
                entity_type=entity['type'],
                document_id=document.id,
                metadata=entity.get('metadata', {}),
                confidence=entity.get('confidence', 0.8)
            )
            stored_entities.append(stored_entity)
        
        for rel in relationships:
            stored_rel = await self.relationship_service.create_relationship(
                source_id=rel['source'],
                target_id=rel['target'],
                relationship_type=rel['type'],
                properties=rel.get('properties', {}),
                confidence=rel.get('confidence', 0.7)
            )
            stored_relationships.append(stored_rel)
        
        return {
            'document_id': document.id,
            'entities_extracted': len(stored_entities),
            'relationships_extracted': len(stored_relationships),
            'chunks_processed': len(chunks),
            'entities': stored_entities,
            'relationships': stored_relationships
        }
    
    async def _extract_entities_from_text(
        self,
        text: str,
        document_id: str,
        chunk_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from text using patterns and NLP
        
        Args:
            text: Text to extract entities from
            document_id: Associated document ID
            chunk_id: Optional chunk ID
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Pattern-based extraction
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                entity_text = match.group().strip()
                
                # Skip common words and short matches
                if len(entity_text) < 3 or entity_text.lower() in ['the', 'and', 'or', 'but']:
                    continue
                
                entity = {
                    'name': entity_text,
                    'type': entity_type,
                    'document_id': document_id,
                    'chunk_id': chunk_id,
                    'position': match.span(),
                    'confidence': 0.7,
                    'metadata': {
                        'extraction_method': 'pattern',
                        'pattern': entity_type
                    }
                }
                entities.append(entity)
        
        # NLP-based extraction if processor available
        if self.nlp_processor:
            nlp_entities = await self._extract_entities_with_nlp(text)
            
            for nlp_entity in nlp_entities:
                nlp_entity['document_id'] = document_id
                nlp_entity['chunk_id'] = chunk_id
                nlp_entity['metadata']['extraction_method'] = 'nlp'
                entities.append(nlp_entity)
        
        # Deduplicate entities
        unique_entities = {}
        for entity in entities:
            key = f"{entity['name'].lower()}_{entity['type']}"
            if key not in unique_entities or entity['confidence'] > unique_entities[key]['confidence']:
                unique_entities[key] = entity
        
        return list(unique_entities.values())
    
    async def _extract_entities_with_nlp(
        self,
        text: str
    ) -> List[Dict[str, Any]]:
        """
        Extract entities using NLP processor
        
        Args:
            text: Text to process
            
        Returns:
            List of NLP-extracted entities
        """
        if not self.nlp_processor:
            return []
        
        try:
            # This would use spaCy, NLTK, or other NLP libraries
            # Simplified placeholder implementation
            entities = []
            
            # Mock NLP entity extraction
            doc = self.nlp_processor.process(text)
            
            for ent in doc.entities:
                entities.append({
                    'name': ent.text,
                    'type': ent.label_.lower(),
                    'confidence': ent.confidence if hasattr(ent, 'confidence') else 0.8,
                    'metadata': {
                        'nlp_model': 'default',
                        'start_char': ent.start_char,
                        'end_char': ent.end_char
                    }
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in NLP entity extraction: {e}")
            return []
    
    async def _extract_relationships_from_text(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        document_id: str,
        chunk_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities from text
        
        Args:
            text: Text to analyze
            entities: List of entities found in text
            document_id: Associated document ID
            chunk_id: Optional chunk ID
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        # Create entity name to ID mapping
        entity_map = {e['name'].lower(): e for e in entities}
        
        # Relationship patterns
        rel_patterns = [
            (r'(\w+)\s+(?:is|was|are|were)\s+(?:a|an|the)?\s*(\w+)', 'is_a'),
            (r'(\w+)\s+(?:has|have|had)\s+(\w+)', 'has'),
            (r'(\w+)\s+(?:works?|worked)\s+(?:at|for|with)\s+(\w+)', 'works_for'),
            (r'(\w+)\s+(?:owns?|owned)\s+(\w+)', 'owns'),
            (r'(\w+)\s+(?:manages?|managed)\s+(\w+)', 'manages'),
            (r'(\w+)\s+(?:created?|creates?|built|builds?)\s+(\w+)', 'created'),
            (r'(\w+)\s+(?:located|based)\s+(?:in|at)\s+(\w+)', 'located_in'),
            (r'(\w+)\s+(?:part of|belongs to|member of)\s+(\w+)', 'part_of')
        ]
        
        # Extract relationships using patterns
        for pattern, rel_type in rel_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                source_text = match.group(1).lower()
                target_text = match.group(2).lower()
                
                # Check if both entities exist
                if source_text in entity_map and target_text in entity_map:
                    relationship = {
                        'source': entity_map[source_text].get('id', source_text),
                        'target': entity_map[target_text].get('id', target_text),
                        'type': rel_type,
                        'confidence': 0.6,
                        'properties': {
                            'document_id': document_id,
                            'chunk_id': chunk_id,
                            'extracted_from': match.group()
                        }
                    }
                    relationships.append(relationship)
        
        # Co-occurrence based relationships
        relationships.extend(
            await self._extract_cooccurrence_relationships(
                text, entities, document_id, chunk_id
            )
        )
        
        return relationships
    
    async def _extract_cooccurrence_relationships(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        document_id: str,
        chunk_id: Optional[str] = None,
        window_size: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships based on entity co-occurrence
        
        Args:
            text: Text to analyze
            entities: List of entities
            document_id: Document ID
            chunk_id: Optional chunk ID
            window_size: Character window for co-occurrence
            
        Returns:
            Co-occurrence based relationships
        """
        relationships = []
        
        # Sort entities by position if available
        positioned_entities = [
            e for e in entities 
            if 'position' in e and e.get('position')
        ]
        
        # Find co-occurring entities
        for i, entity1 in enumerate(positioned_entities):
            pos1_start, pos1_end = entity1['position']
            
            for entity2 in positioned_entities[i+1:]:
                pos2_start, pos2_end = entity2['position']
                
                # Check if entities are within window
                distance = min(
                    abs(pos2_start - pos1_end),
                    abs(pos1_start - pos2_end)
                )
                
                if distance <= window_size:
                    # Extract context between entities
                    context_start = min(pos1_end, pos2_end)
                    context_end = max(pos1_start, pos2_start)
                    context = text[context_start:context_end] if context_start < context_end else ""
                    
                    # Determine relationship type from context
                    rel_type = self._infer_relationship_type(context)
                    
                    relationship = {
                        'source': entity1.get('id', entity1['name']),
                        'target': entity2.get('id', entity2['name']),
                        'type': rel_type,
                        'confidence': 0.5 * (1 - distance / window_size),  # Closer = higher confidence
                        'properties': {
                            'document_id': document_id,
                            'chunk_id': chunk_id,
                            'distance': distance,
                            'co_occurrence': True
                        }
                    }
                    relationships.append(relationship)
        
        return relationships
    
    def _infer_relationship_type(self, context: str) -> str:
        """
        Infer relationship type from context text
        
        Args:
            context: Text between two entities
            
        Returns:
            Inferred relationship type
        """
        context_lower = context.lower()
        
        # Simple keyword-based inference
        if any(word in context_lower for word in ['work', 'employ', 'job']):
            return 'professional'
        elif any(word in context_lower for word in ['own', 'possess', 'belong']):
            return 'ownership'
        elif any(word in context_lower for word in ['create', 'build', 'develop', 'make']):
            return 'created'
        elif any(word in context_lower for word in ['manage', 'lead', 'direct', 'supervise']):
            return 'manages'
        elif any(word in context_lower for word in ['locate', 'based', 'headquarter']):
            return 'location'
        elif any(word in context_lower for word in ['partner', 'collaborate', 'associate']):
            return 'partnership'
        elif any(word in context_lower for word in ['compete', 'rival', 'versus']):
            return 'competition'
        else:
            return 'related'
    
    async def _merge_similar_entities(
        self,
        entities: List[Dict[str, Any]],
        similarity_threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Merge similar entities based on name similarity
        
        Args:
            entities: List of entities to merge
            similarity_threshold: Minimum similarity for merging
            
        Returns:
            Merged entity list
        """
        merged = []
        processed = set()
        
        for i, entity1 in enumerate(entities):
            if i in processed:
                continue
            
            # Find similar entities
            similar_group = [entity1]
            
            for j, entity2 in enumerate(entities[i+1:], start=i+1):
                if j in processed:
                    continue
                
                similarity = self._calculate_similarity(
                    entity1['name'],
                    entity2['name']
                )
                
                if similarity >= similarity_threshold and entity1['type'] == entity2['type']:
                    similar_group.append(entity2)
                    processed.add(j)
            
            # Merge the group
            if len(similar_group) > 1:
                merged_entity = self._merge_entity_group(similar_group)
                merged.append(merged_entity)
            else:
                merged.append(entity1)
            
            processed.add(i)
        
        return merged
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Simple Jaccard similarity
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_entity_group(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge a group of similar entities
        
        Args:
            entities: List of similar entities
            
        Returns:
            Merged entity
        """
        # Use the most common or longest name
        names = [e['name'] for e in entities]
        merged_name = max(names, key=lambda x: (names.count(x), len(x)))
        
        # Merge metadata
        merged_metadata = {}
        for entity in entities:
            merged_metadata.update(entity.get('metadata', {}))
        
        merged_metadata['merged_from'] = [e['name'] for e in entities]
        merged_metadata['merge_count'] = len(entities)
        
        # Use highest confidence
        max_confidence = max(e.get('confidence', 0.5) for e in entities)
        
        return {
            'name': merged_name,
            'type': entities[0]['type'],
            'document_id': entities[0]['document_id'],
            'confidence': min(1.0, max_confidence * 1.1),  # Boost confidence slightly
            'metadata': merged_metadata
        }
    
    async def update_graph_statistics(
        self,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update and return graph statistics
        
        Args:
            document_id: Optional document filter
            
        Returns:
            Graph statistics
        """
        stats = {
            'total_entities': 0,
            'total_relationships': 0,
            'entity_types': {},
            'relationship_types': {},
            'avg_entity_connections': 0,
            'graph_density': 0,
            'connected_components': 0
        }
        
        if not self.graph_store:
            return stats
        
        # Get entity statistics
        entity_query = "MATCH (n) "
        if document_id:
            entity_query += f"WHERE n.document_id = '{document_id}' "
        entity_query += "RETURN count(n) as count, labels(n) as types"
        
        entity_results = await self.graph_store.execute_query(entity_query)
        
        for result in entity_results:
            stats['total_entities'] += result['count']
            for entity_type in result.get('types', []):
                stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1
        
        # Get relationship statistics
        rel_query = "MATCH ()-[r]-() "
        if document_id:
            rel_query += f"WHERE r.document_id = '{document_id}' "
        rel_query += "RETURN count(r) as count, type(r) as type"
        
        rel_results = await self.graph_store.execute_query(rel_query)
        
        for result in rel_results:
            stats['total_relationships'] += result['count']
            rel_type = result['type']
            stats['relationship_types'][rel_type] = stats['relationship_types'].get(rel_type, 0) + result['count']
        
        # Calculate average connections
        if stats['total_entities'] > 0:
            stats['avg_entity_connections'] = (
                2 * stats['total_relationships'] / stats['total_entities']
            )
            
            # Calculate graph density
            max_edges = stats['total_entities'] * (stats['total_entities'] - 1) / 2
            if max_edges > 0:
                stats['graph_density'] = stats['total_relationships'] / max_edges
        
        return stats
    
    async def validate_graph(
        self,
        fix_issues: bool = False
    ) -> Dict[str, Any]:
        """
        Validate graph integrity and optionally fix issues
        
        Args:
            fix_issues: Whether to attempt fixing found issues
            
        Returns:
            Validation results
        """
        issues = []
        fixes_applied = []
        
        if not self.graph_store:
            return {'issues': ['No graph store available'], 'fixes_applied': []}
        
        # Check for orphaned nodes
        orphan_query = """
        MATCH (n)
        WHERE NOT (n)-[]-()  
        RETURN n.id as id, labels(n) as types
        """
        
        orphans = await self.graph_store.execute_query(orphan_query)
        
        if orphans:
            issues.append(f"Found {len(orphans)} orphaned nodes")
            
            if fix_issues:
                # Could delete or connect orphans
                for orphan in orphans:
                    logger.info(f"Orphaned node: {orphan['id']}")
                fixes_applied.append(f"Logged {len(orphans)} orphaned nodes")
        
        # Check for duplicate entities
        duplicate_query = """
        MATCH (n)
        WITH n.name as name, n.type as type, collect(n) as nodes
        WHERE size(nodes) > 1
        RETURN name, type, size(nodes) as count
        """
        
        duplicates = await self.graph_store.execute_query(duplicate_query)
        
        if duplicates:
            issues.append(f"Found {len(duplicates)} duplicate entity groups")
            
            if fix_issues:
                # Could merge duplicates
                for dup in duplicates:
                    logger.info(f"Duplicate entities: {dup['name']} ({dup['count']} instances)")
                fixes_applied.append(f"Identified {len(duplicates)} duplicate groups for merging")
        
        # Check for self-loops
        self_loop_query = """
        MATCH (n)-[r]-(n)
        RETURN n.id as id, type(r) as rel_type
        """
        
        self_loops = await self.graph_store.execute_query(self_loop_query)
        
        if self_loops:
            issues.append(f"Found {len(self_loops)} self-loops")
            
            if fix_issues:
                for loop in self_loops:
                    logger.info(f"Self-loop on node: {loop['id']}")
                fixes_applied.append(f"Logged {len(self_loops)} self-loops")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'fixes_applied': fixes_applied,
            'timestamp': datetime.now().isoformat()
        }