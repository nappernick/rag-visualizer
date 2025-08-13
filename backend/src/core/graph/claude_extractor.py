"""
Claude-based entity and relationship extraction for knowledge graph construction
"""
import os
import json
import logging
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import boto3
from botocore.exceptions import ClientError

from models.schemas import Chunk, Entity, Relationship

logger = logging.getLogger(__name__)


@dataclass
class EntityRecord:
    name: str
    entity_type: str
    chunk_ids: List[str] = field(default_factory=list)
    frequency: int = 1
    metadata: Dict = field(default_factory=dict)


@dataclass  
class RelationRecord:
    source: str
    target: str
    relation_type: str = "RELATES_TO"
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)


class ClaudeGraphExtractor:
    """Uses Claude via AWS Bedrock for advanced entity and relationship extraction"""
    
    def __init__(self):
        """Initialize Claude client"""
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-2')
        )
        # Use Claude Sonnet 4 for entity extraction with US inference profile
        self.model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
        
    def extract_from_chunks(self, chunks: List[Chunk], 
                           document_id: str) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from chunks using Claude"""
        entity_map: Dict[str, EntityRecord] = {}
        relationships_list = []
        
        # Combine all chunks into a single document text for better context
        full_text = "\n\n".join([chunk.content for chunk in chunks])
        
        # Extract entities and relationships from the full document
        extraction_result = self._extract_with_claude(full_text)
        
        if extraction_result:
            # Process entities
            for entity_data in extraction_result.get('entities', []):
                entity_name = entity_data['name']
                entity_type = entity_data['type']
                
                key = entity_name.lower()
                if key not in entity_map:
                    entity_map[key] = EntityRecord(
                        name=entity_name,
                        entity_type=entity_type,
                        chunk_ids=[chunk.id for chunk in chunks],  # All chunks
                        metadata=entity_data.get('metadata', {})
                    )
                else:
                    entity_map[key].frequency += 1
            
            # Process relationships
            for rel_data in extraction_result.get('relationships', []):
                relationships_list.append(RelationRecord(
                    source=rel_data['source'],
                    target=rel_data['target'],
                    relation_type=rel_data['type'],
                    weight=rel_data.get('confidence', 1.0),
                    metadata=rel_data.get('metadata', {})
                ))
        
        # Convert to output format
        entities = []
        for record in entity_map.values():
            entity = Entity(
                name=record.name,
                entity_type=record.entity_type,
                document_ids=[document_id],
                chunk_ids=record.chunk_ids,
                frequency=record.frequency,
                metadata=record.metadata
            )
            entities.append(entity)
        
        # Convert relationships
        relationships = []
        for rel_record in relationships_list:
            # Find entity IDs
            source_entity = next((e for e in entities if e.name.lower() == rel_record.source.lower()), None)
            target_entity = next((e for e in entities if e.name.lower() == rel_record.target.lower()), None)
            
            if source_entity and target_entity:
                relationship = Relationship(
                    source_entity_id=source_entity.id,
                    target_entity_id=target_entity.id,
                    relationship_type=rel_record.relation_type,
                    weight=rel_record.weight,
                    document_ids=[document_id],
                    metadata=rel_record.metadata
                )
                relationships.append(relationship)
        
        # Sort by relevance
        entities.sort(key=lambda e: e.frequency, reverse=True)
        relationships.sort(key=lambda r: r.weight, reverse=True)
        
        return entities, relationships
    
    def _extract_with_claude(self, text: str) -> Optional[Dict]:
        """Use Claude to extract entities and relationships from text"""
        
        prompt = """You are an expert knowledge graph extractor. Analyze the following text and extract:
1. Key entities (people, organizations, technologies, concepts, etc.)
2. Relationships between these entities

For the RAG (Retrieval-Augmented Generation) domain, pay special attention to:
- AI/ML models and frameworks (GPT, BERT, LangChain, etc.)
- Databases and vector stores (PostgreSQL, Pinecone, Qdrant, etc.)
- Cloud services (AWS, Azure, GCP, etc.)
- Algorithms and techniques (HNSW, embeddings, chunking, etc.)
- Programming languages and libraries
- Key concepts and patterns

CRITICAL REQUIREMENTS:
1. Create a FULLY CONNECTED knowledge graph - every entity must be connected to at least one other entity
2. NO orphaned entities or isolated subgraphs - ensure all parts of the graph are interconnected
3. If an entity seems isolated, find or create a logical relationship to connect it to the main graph
4. Use broad relationships like "RELATES_TO" or "ASSOCIATED_WITH" if specific relationships are unclear
5. Prioritize creating a cohesive, navigable graph structure over perfect precision

Return your response as a JSON object with this structure:
{
  "entities": [
    {
      "name": "Entity Name",
      "type": "entity_type",  // e.g., "person", "organization", "technology", "concept", "model", "database", "framework"
      "metadata": {}  // Optional additional properties
    }
  ],
  "relationships": [
    {
      "source": "Source Entity Name",
      "target": "Target Entity Name", 
      "type": "RELATIONSHIP_TYPE",  // e.g., "USES", "IMPLEMENTS", "CONTAINS", "DEPENDS_ON", "IMPROVES", "INTEGRATES", "RELATES_TO"
      "confidence": 0.95,  // Confidence score 0-1
      "metadata": {}  // Optional additional properties
    }
  ]
}

IMPORTANT: Ensure EVERY entity appears in at least one relationship. Create logical connections between all entities to form a single, connected graph.

Text to analyze:
"""
        
        # Retry logic for throttling
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Prepare the request
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4000,
                    "temperature": 0.1,  # Low temperature for consistent extraction
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt + text[:8000]  # Limit text length
                        }
                    ],
                    "top_p": 0.9,
                })
                
                # Invoke Claude
                response = self.bedrock_runtime.invoke_model(
                    body=body,
                    modelId=self.model_id,
                    accept='application/json',
                    contentType='application/json'
                )
                
                # Parse response after successful invocation
                response_body = json.loads(response['body'].read())
                claude_response = response_body.get('content', [{}])[0].get('text', '')
                break  # Success, exit retry loop
                
            except ClientError as e:
                if 'ThrottlingException' in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Throttling error, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    logger.error(f"AWS Bedrock error: {e}")
                    return None
            
        # Extract JSON from response
        try:
            # Find JSON in the response
            json_start = claude_response.find('{')
            json_end = claude_response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = claude_response[json_start:json_end]
                result = json.loads(json_str)
                
                logger.info(f"Claude extracted {len(result.get('entities', []))} entities and {len(result.get('relationships', []))} relationships")
                return result
            else:
                logger.warning("No valid JSON found in Claude response")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            logger.debug(f"Response was: {claude_response[:500]}...")
            return None
        except Exception as e:
            logger.error(f"Error processing Claude response: {e}")
            return None