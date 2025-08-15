"""
Demo API endpoints for showcasing RAG system capabilities
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
import json
import logging
import boto3
from botocore.exceptions import ClientError

from ..db import get_session
from ..core.query.query_enhancer import QueryEnhancer, QueryType
from ..services.vector_service import get_vector_service
from ..services.graph_service import get_graph_service
from ..services.embedding_service import get_embedding_service
from ..services.storage import get_storage_service
from ..core.scoring.match_scorer import MatchScorer
from ..core.scoring.fusion_ranker import FusionRanker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/demo", tags=["demo"])

# Initialize Bedrock client
try:
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-west-2'
    )
    logger.info("Bedrock client initialized for demo API")
except Exception as e:
    logger.warning(f"Could not initialize Bedrock client: {e}")
    bedrock_client = None


class DemoSearchRequest(BaseModel):
    query: str
    mode: str = "smart"  # smart, vector, graph, hybrid
    includeExplanations: bool = True
    includeDecomposition: bool = False
    maxResults: int = 10


class DemoSearchResponse(BaseModel):
    results: List[Dict]
    decomposition: Optional[Dict] = None
    total_results: int
    retrieval_strategy: str
    processing_time_ms: float
    average_confidence: float
    metadata: Dict = {}


class SuggestionRequest(BaseModel):
    query: str
    max_suggestions: int = 5


class SummarizeRequest(BaseModel):
    document_id: str
    style: str = "brief"  # brief, detailed, technical


class ExploreRequest(BaseModel):
    entity_ids: List[str]
    max_hops: int = 2
    limit: int = 10


class PathFindRequest(BaseModel):
    start_entity: str
    end_entity: str
    max_hops: int = 3


class AnalyzeRequest(BaseModel):
    document_ids: List[str]
    analysis_type: str = "comprehensive"  # comprehensive, summary, entities, relationships


@router.post("/search", response_model=DemoSearchResponse)
async def enhanced_search(
    request: DemoSearchRequest,
    db: Session = Depends(get_session),
    embedding_service=Depends(get_embedding_service),
    vector_service=Depends(get_vector_service),
    graph_service=Depends(get_graph_service),
    storage=Depends(get_storage_service)
):
    """
    Enhanced search endpoint with explanations and decomposition
    """
    start_time = datetime.now()
    
    try:
        # Initialize query enhancer
        query_enhancer = QueryEnhancer(bedrock_client=bedrock_client, use_local_models=False)
        
        # Decompose query if requested
        decomposition = None
        if request.includeDecomposition:
            enhanced_query = await query_enhancer.enhance_query(
                query=request.query,
                max_sub_queries=5
            )
            decomposition = {
                "original": enhanced_query.original,
                "sub_queries": [
                    {
                        "question": sq.question,
                        "type": sq.type.value,
                        "dependencies": sq.dependencies,
                        "priority": sq.priority,
                        "keywords": sq.keywords
                    }
                    for sq in enhanced_query.sub_queries
                ],
                "reasoning_path": enhanced_query.reasoning_path,
                "complexity_score": enhanced_query.complexity_score,
                "query_type": enhanced_query.query_type.value
            }
        
        # Determine retrieval strategy based on mode
        if request.mode == "smart":
            # Use Claude Sonnet 3.5 for intelligent strategy selection
            strategy = await _determine_smart_strategy(request.query, bedrock_client)
        else:
            strategy = request.mode
        
        # Execute search based on strategy
        results = []
        
        if strategy in ["vector", "hybrid"]:
            # Generate embedding
            query_embedding = await embedding_service.generate_embedding(request.query)
            
            if query_embedding:
                # Search vectors
                vector_results = await vector_service.search_similar(
                    query_embedding=query_embedding,
                    limit=request.maxResults * 2,
                    score_threshold=0.0
                )
                
                for i, result in enumerate(vector_results[:request.maxResults]):
                    # Get document info
                    doc = await storage.get_document(result["document_id"])
                    
                    # Calculate match score
                    match_score = MatchScorer.calculate_vector_match_score(
                        query=request.query,
                        content=result["content"],
                        cosine_similarity=result["score"],
                        metadata=result.get("metadata", {})
                    )
                    
                    # Generate explanation if requested
                    explanation = None
                    if request.includeExplanations:
                        explanation = await _generate_explanation(
                            query=request.query,
                            content=result["content"],
                            score=match_score,
                            source="vector",
                            bedrock_client=bedrock_client
                        )
                    
                    results.append({
                        "id": f"vector_{i}",
                        "content": result["content"],
                        "score": match_score / 100,  # Convert to 0-1 scale
                        "source": "vector",
                        "document_id": result["document_id"],
                        "document_title": doc.get("title", "Unknown") if doc else "Unknown",
                        "chunk_id": result["chunk_id"],
                        "metadata": {
                            "chunk_type": result["chunk_type"],
                            "cosine_similarity": result["score"]
                        },
                        "explanation": explanation,
                        "highlights": _extract_highlights(request.query, result["content"])
                    })
        
        if strategy in ["graph", "hybrid"]:
            # Graph retrieval
            graph_entities = await graph_service.production_graph_retrieval(
                query=request.query,
                limit=request.maxResults,
                max_hops=2
            )
            
            for i, entity in enumerate(graph_entities):
                match_score = MatchScorer.calculate_graph_match_score(
                    query=request.query,
                    entity_name=entity["name"],
                    entity_type=entity["entity_type"],
                    match_type=entity.get("match_type", "exact"),
                    confidence=entity.get("confidence", 0.5),
                    path_length=entity.get("path_length", 0)
                )
                
                # Generate explanation if requested
                explanation = None
                if request.includeExplanations:
                    explanation = await _generate_explanation(
                        query=request.query,
                        content=f"Entity: {entity['name']} (Type: {entity['entity_type']})",
                        score=match_score,
                        source="graph",
                        bedrock_client=bedrock_client
                    )
                
                results.append({
                    "id": f"graph_{i}",
                    "content": f"Entity: {entity['name']} (Type: {entity['entity_type']})",
                    "score": match_score / 100,
                    "source": "graph",
                    "metadata": {
                        "entity_id": entity["id"],
                        "entity_type": entity["entity_type"],
                        "frequency": entity["frequency"],
                        "match_type": entity.get("match_type", "exact")
                    },
                    "explanation": explanation,
                    "highlights": [entity["name"]]
                })
        
        # Calculate average confidence
        avg_confidence = sum(r["score"] for r in results) / len(results) if results else 0
        
        # Sort results by score
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:request.maxResults]
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return DemoSearchResponse(
            results=results,
            decomposition=decomposition,
            total_results=len(results),
            retrieval_strategy=strategy,
            processing_time_ms=processing_time,
            average_confidence=avg_confidence,
            metadata={
                "mode": request.mode,
                "decomposed": request.includeDecomposition,
                "explained": request.includeExplanations
            }
        )
        
    except Exception as e:
        logger.error(f"Demo search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/suggest")
async def get_suggestions(
    request: SuggestionRequest,
    db: Session = Depends(get_session)
):
    """
    Get query suggestions using Claude Haiku 3.5
    """
    try:
        if not bedrock_client:
            # Return simple suggestions without LLM
            return {"suggestions": _get_simple_suggestions(request.query)}
        
        # Use Claude Haiku 3.5 for quick suggestions
        prompt = f"""Given the partial query: "{request.query}"
        
Suggest {request.max_suggestions} relevant complete queries for a RAG system.
Return ONLY a JSON array of strings, no other text:
["suggestion1", "suggestion2", ...]"""

        response = bedrock_client.invoke_model(
            modelId='us.anthropic.claude-3-5-haiku-20241022-v1:0',
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                'messages': [{'role': 'user', 'content': prompt}],
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 200,
                'temperature': 0.7
            })
        )
        
        result = json.loads(response['body'].read())
        content = result.get('content', [{}])[0].get('text', '[]')
        suggestions = json.loads(content)
        
        return {"suggestions": suggestions[:request.max_suggestions]}
        
    except Exception as e:
        logger.error(f"Suggestion error: {e}")
        return {"suggestions": _get_simple_suggestions(request.query)}


@router.post("/decompose")
async def decompose_query(
    query: str,
    db: Session = Depends(get_session)
):
    """
    Decompose complex query using Claude Sonnet 3.5
    """
    try:
        if not bedrock_client:
            raise HTTPException(status_code=503, detail="Bedrock client not available")
        
        # Use Claude Sonnet 4 for complex decomposition
        prompt = f"""Decompose this complex query into simpler sub-questions for a RAG system.
Each sub-question should be independently answerable.

Query: "{query}"

Return a JSON object with this structure:
{{
  "sub_queries": [
    {{
      "question": "sub-question text",
      "type": "factual|analytical|comparative|exploratory|navigational|multi_hop",
      "dependencies": [indices of questions this depends on],
      "priority": "high|medium|low",
      "keywords": ["key", "terms"]
    }}
  ],
  "reasoning_path": "Step-by-step reasoning description",
  "complexity_score": 0.0-1.0
}}"""

        response = bedrock_client.invoke_model(
            modelId='us.anthropic.claude-sonnet-4-20250514-v1:0',
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                'messages': [{'role': 'user', 'content': prompt}],
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 1000,
                'temperature': 0.3
            })
        )
        
        result = json.loads(response['body'].read())
        content = result.get('content', [{}])[0].get('text', '{}')
        decomposition = json.loads(content)
        
        return decomposition
        
    except Exception as e:
        logger.error(f"Decomposition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize")
async def summarize_document(
    request: SummarizeRequest,
    db: Session = Depends(get_session),
    storage=Depends(get_storage_service)
):
    """
    Summarize document using Claude Haiku 3.5 for speed
    """
    try:
        # Get document
        document = await storage.get_document(request.document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not bedrock_client:
            # Return simple summary without LLM
            return {
                "summary": document["content"][:500] + "..." if len(document["content"]) > 500 else document["content"],
                "key_points": [],
                "entities": []
            }
        
        # Determine prompt based on style
        if request.style == "brief":
            prompt = f"""Provide a brief 2-3 sentence summary of this document:

{document['content'][:3000]}

Return JSON:
{{
  "summary": "brief summary",
  "key_points": ["point1", "point2", "point3"]
}}"""
        elif request.style == "detailed":
            prompt = f"""Provide a detailed summary of this document:

{document['content'][:5000]}

Return JSON:
{{
  "summary": "detailed summary",
  "key_points": ["point1", "point2", ...],
  "main_topics": ["topic1", "topic2", ...],
  "entities": ["entity1", "entity2", ...]
}}"""
        else:  # technical
            prompt = f"""Provide a technical summary of this document:

{document['content'][:4000]}

Return JSON:
{{
  "summary": "technical summary",
  "technical_concepts": ["concept1", "concept2", ...],
  "methodologies": ["method1", "method2", ...],
  "key_findings": ["finding1", "finding2", ...]
}}"""
        
        # Use Claude Haiku 3.5 for quick summarization
        response = bedrock_client.invoke_model(
            modelId='us.anthropic.claude-3-5-haiku-20241022-v1:0',
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                'messages': [{'role': 'user', 'content': prompt}],
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 500,
                'temperature': 0.3
            })
        )
        
        result = json.loads(response['body'].read())
        content = result.get('content', [{}])[0].get('text', '{}')
        summary_data = json.loads(content)
        
        return {
            "document_id": request.document_id,
            "document_title": document.get("title", "Unknown"),
            "style": request.style,
            **summary_data
        }
        
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explore")
async def explore_graph(
    request: ExploreRequest,
    db: Session = Depends(get_session),
    graph_service=Depends(get_graph_service)
):
    """
    Multi-hop graph exploration using Claude Sonnet 3.5 for reasoning
    """
    try:
        # Perform simple graph traversal for related entities
        exploration_results = []
        
        # Check if Neo4j is available
        if not graph_service.initialized:
            # Fallback: Return mock data for demo purposes
            for entity_id in request.entity_ids[:1]:
                exploration_results.append({
                    "id": f"related_{entity_id}_1",
                    "name": f"Related Entity 1",
                    "entity_type": "concept",
                    "frequency": 5
                })
                exploration_results.append({
                    "id": f"related_{entity_id}_2", 
                    "name": f"Related Entity 2",
                    "entity_type": "technology",
                    "frequency": 3
                })
        else:
            for entity_id in request.entity_ids[:3]:  # Limit to 3 seed entities
                # Get related entities
                related = await graph_service.get_related_entities(
                    entity_id=entity_id,
                    max_hops=request.max_hops
                )
                
                # Convert to expected format
                for entity, distance in related[:request.limit]:
                    exploration_results.append(entity)
        
        # Use Claude Sonnet 4 to analyze and explain the exploration
        if bedrock_client and exploration_results:
            entities_desc = "\n".join([
                f"- {e.get('name', 'Unknown')} ({e.get('entity_type', 'Unknown')})"
                for e in exploration_results[:10]
            ])
            
            prompt = f"""Analyze these graph exploration results and explain the relationships:

Entities found:
{entities_desc}

Provide insights about:
1. Key relationships discovered
2. Potential knowledge patterns
3. Suggested next exploration steps

Return JSON:
{{
  "insights": "main insights",
  "patterns": ["pattern1", "pattern2"],
  "next_steps": ["suggestion1", "suggestion2"]
}}"""

            response = bedrock_client.invoke_model(
                modelId='us.anthropic.claude-sonnet-4-20250514-v1:0',
                contentType='application/json',
                accept='application/json',
                body=json.dumps({
                    'messages': [{'role': 'user', 'content': prompt}],
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 500,
                    'temperature': 0.5
                })
            )
            
            result = json.loads(response['body'].read())
            content = result.get('content', [{}])[0].get('text', '{}')
            analysis = json.loads(content)
        else:
            analysis = {
                "insights": "Graph exploration completed",
                "patterns": [],
                "next_steps": []
            }
        
        return {
            "entities": exploration_results[:request.limit],
            "total_found": len(exploration_results),
            "max_hops": request.max_hops,
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Exploration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/find-path")
async def find_path(
    request: PathFindRequest,
    db: Session = Depends(get_session),
    graph_service=Depends(get_graph_service)
):
    """
    Find path between entities in the knowledge graph
    """
    try:
        # This would use Neo4j's pathfinding capabilities
        # For now, return a mock response
        return {
            "start": request.start_entity,
            "end": request.end_entity,
            "paths": [
                {
                    "length": 2,
                    "nodes": [request.start_entity, "intermediate_entity", request.end_entity],
                    "relationships": ["relates_to", "connected_with"]
                }
            ],
            "shortest_path_length": 2
        }
        
    except Exception as e:
        logger.error(f"Path finding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze_documents(
    request: AnalyzeRequest,
    db: Session = Depends(get_session),
    storage=Depends(get_storage_service)
):
    """
    Deep analysis of documents using Claude Sonnet 3.5
    """
    try:
        if not bedrock_client:
            raise HTTPException(status_code=503, detail="Bedrock client not available")
        
        # Get documents
        documents = []
        for doc_id in request.document_ids[:3]:  # Limit to 3 documents
            doc = await storage.get_document(doc_id)
            if doc:
                documents.append(doc)
        
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found")
        
        # Prepare content for analysis
        docs_content = "\n\n---\n\n".join([
            f"Document: {doc.get('title', 'Unknown')}\n{doc.get('content', '')[:2000]}"
            for doc in documents
        ])
        
        # Use Claude Sonnet 4 for comprehensive analysis
        prompt = f"""Perform a {request.analysis_type} analysis of these documents:

{docs_content}

Provide:
1. Main themes and topics
2. Key entities and concepts
3. Relationships between documents
4. Insights and recommendations

Return JSON:
{{
  "themes": ["theme1", "theme2"],
  "key_concepts": ["concept1", "concept2"],
  "entities": [{{"name": "entity", "type": "type", "importance": "high|medium|low"}}],
  "relationships": ["relationship1", "relationship2"],
  "insights": "main insights",
  "recommendations": ["rec1", "rec2"]
}}"""

        response = bedrock_client.invoke_model(
            modelId='us.anthropic.claude-sonnet-4-20250514-v1:0',
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                'messages': [{'role': 'user', 'content': prompt}],
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 1500,
                'temperature': 0.5
            })
        )
        
        result = json.loads(response['body'].read())
        content = result.get('content', [{}])[0].get('text', '{}')
        analysis = json.loads(content)
        
        return {
            "document_ids": request.document_ids,
            "analysis_type": request.analysis_type,
            "document_count": len(documents),
            **analysis
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions

async def _determine_smart_strategy(query: str, bedrock_client) -> str:
    """Use Claude Sonnet 4 to determine optimal search strategy"""
    if not bedrock_client:
        return "hybrid"
    
    try:
        prompt = f"""Determine the best search strategy for this query:
Query: "{query}"

Options:
- vector: Best for semantic similarity, concepts, and general questions
- graph: Best for entities, relationships, and specific facts
- hybrid: Best for complex queries needing both

Return only one word: vector, graph, or hybrid"""

        response = bedrock_client.invoke_model(
            modelId='us.anthropic.claude-sonnet-4-20250514-v1:0',
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                'messages': [{'role': 'user', 'content': prompt}],
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 10,
                'temperature': 0.1
            })
        )
        
        result = json.loads(response['body'].read())
        strategy = result.get('content', [{}])[0].get('text', 'hybrid').strip().lower()
        
        if strategy not in ['vector', 'graph', 'hybrid']:
            strategy = 'hybrid'
        
        return strategy
        
    except Exception:
        return "hybrid"


async def _generate_explanation(
    query: str,
    content: str,
    score: float,
    source: str,
    bedrock_client
) -> Optional[str]:
    """Generate explanation for why a result matches using Claude Haiku 3.5"""
    if not bedrock_client:
        return None
    
    try:
        prompt = f"""Explain in one sentence why this result matches the query:
Query: "{query}"
Result: "{content[:200]}"
Match Score: {score:.0f}%
Source: {source}

Provide a brief, clear explanation."""

        response = bedrock_client.invoke_model(
            modelId='us.anthropic.claude-3-5-haiku-20241022-v1:0',
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                'messages': [{'role': 'user', 'content': prompt}],
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 100,
                'temperature': 0.3
            })
        )
        
        result = json.loads(response['body'].read())
        explanation = result.get('content', [{}])[0].get('text', '').strip()
        
        return explanation
        
    except Exception:
        return None


def _extract_highlights(query: str, content: str) -> List[str]:
    """Extract key phrases from content that match the query"""
    query_words = query.lower().split()
    content_lower = content.lower()
    highlights = []
    
    # Find sentences containing query words
    sentences = content.split('.')
    for sentence in sentences[:5]:  # Limit to first 5 sentences
        sentence_lower = sentence.lower()
        if any(word in sentence_lower for word in query_words):
            # Extract key phrase around the matching word
            words = sentence.split()
            if len(words) > 5:
                highlights.append(' '.join(words[:10]) + '...')
            else:
                highlights.append(sentence.strip())
    
    return highlights[:3]  # Return top 3 highlights


def _get_simple_suggestions(query: str) -> List[str]:
    """Get simple query suggestions without LLM"""
    suggestions = []
    query_lower = query.lower()
    
    # Common query patterns
    if "how" in query_lower:
        suggestions.extend([
            f"{query} work?",
            f"{query} implemented?",
            f"{query} configured?"
        ])
    elif "what" in query_lower:
        suggestions.extend([
            f"{query} is?",
            f"{query} are the benefits?",
            f"{query} are the components?"
        ])
    else:
        suggestions.extend([
            f"How does {query} work?",
            f"What is {query}?",
            f"Explain {query}"
        ])
    
    return suggestions[:5]