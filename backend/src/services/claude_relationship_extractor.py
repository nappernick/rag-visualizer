"""
Claude-based relationship extraction service
"""
import os
import json
import hashlib
from typing import List, Dict, Tuple
import logging
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class ClaudeRelationshipExtractor:
    """Extract relationships between entities using Claude AI"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("No API key found for Claude")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = "claude-3-haiku-20240307"  # Use Haiku for cost efficiency
    
    async def extract_relationships(
        self,
        entities: List[Dict],
        document_text: str,
        document_id: str,
        max_relationships: int = 50
    ) -> List[Dict]:
        """
        Extract relationships between entities using Claude
        
        Args:
            entities: List of entity dictionaries with id, name, and type
            document_text: Full document text
            document_id: Document ID
            max_relationships: Maximum number of relationships to extract
            
        Returns:
            List of relationship dictionaries
        """
        
        if not entities or len(entities) < 2:
            return []
        
        # Prepare entity list for Claude
        entity_list = []
        entity_map = {}
        for entity in entities[:100]:  # Limit to 100 entities for context
            entity_id = entity["id"]
            entity_name = entity["name"]
            entity_type = entity.get("entity_type", "UNKNOWN")
            entity_list.append(f"- {entity_name} ({entity_type}) [ID: {entity_id}]")
            entity_map[entity_name.lower()] = entity_id
        
        # Create prompt for Claude
        prompt = f"""Given the following document and list of entities, identify the most important and meaningful relationships between entities.

ENTITIES (format: Name (Type) [ID: entity_id]):
{chr(10).join(entity_list)}

DOCUMENT:
{document_text[:8000]}  # Limit document length for context

TASK:
1. Identify relationships between the entities listed above that are explicitly or strongly implied in the document
2. Focus on meaningful, specific relationships (not just co-occurrence)
3. Return up to {max_relationships} most important relationships

OUTPUT FORMAT:
Return a JSON array of relationships. Each relationship should have:
{{
  "source": "Entity Name 1",
  "target": "Entity Name 2", 
  "type": "RELATIONSHIP_TYPE",
  "confidence": 0.8,
  "evidence": "Brief quote or reason from document"
}}

Relationship types to use:
- USES (X uses Y)
- IMPLEMENTS (X implements Y)
- EXTENDS (X extends Y)
- CONTAINS (X contains Y)
- DEPENDS_ON (X depends on Y)
- CREATES (X creates Y)
- MANAGES (X manages Y)
- CONNECTS_TO (X connects to Y)
- PART_OF (X is part of Y)
- SIMILAR_TO (X is similar to Y)
- COMPETES_WITH (X competes with Y)
- REPLACES (X replaces Y)

Only return the JSON array, no other text."""

        try:
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse response
            response_text = response.content[0].text.strip()
            
            # Extract JSON from response (handle potential markdown formatting)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            # Parse JSON
            try:
                claude_relationships = json.loads(response_text)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Claude response as JSON: {response_text[:200]}")
                return []
            
            # Convert to our format
            relationships = []
            for rel in claude_relationships:
                source_name = rel.get("source", "").lower()
                target_name = rel.get("target", "").lower()
                
                # Find entity IDs
                source_id = entity_map.get(source_name)
                target_id = entity_map.get(target_name)
                
                if not source_id or not target_id or source_id == target_id:
                    continue
                
                # Create relationship ID
                rel_type = rel.get("type", "RELATED_TO")
                rel_id = hashlib.md5(
                    f"{source_id}_{target_id}_{rel_type}".encode()
                ).hexdigest()[:12]
                
                relationship = {
                    "id": rel_id,
                    "source_entity_id": source_id,
                    "target_entity_id": target_id,
                    "relationship_type": rel_type,
                    "weight": rel.get("confidence", 0.7),
                    "document_ids": [document_id],
                    "metadata": {
                        "extraction_method": "claude",
                        "confidence": rel.get("confidence", 0.7),
                        "evidence": rel.get("evidence", "")[:200],
                        "model": self.model
                    }
                }
                
                relationships.append(relationship)
            
            logger.info(f"Claude extracted {len(relationships)} relationships")
            return relationships
            
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return []