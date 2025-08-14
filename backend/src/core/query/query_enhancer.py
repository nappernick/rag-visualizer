"""
Query Enhancement module for decomposition, expansion, and optimization.
Breaks down complex queries into sub-questions and expands with synonyms and variations.
"""
import asyncio
import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries for classification"""
    FACTUAL = "factual"  # Simple fact-finding
    ANALYTICAL = "analytical"  # Requires analysis or reasoning
    COMPARATIVE = "comparative"  # Comparing multiple items
    EXPLORATORY = "exploratory"  # Open-ended exploration
    NAVIGATIONAL = "navigational"  # Finding specific documents/sections
    MULTI_HOP = "multi_hop"  # Requires connecting multiple pieces


@dataclass
class SubQuery:
    """Represents a decomposed sub-query"""
    question: str
    type: QueryType
    dependencies: List[int] = field(default_factory=list)  # Indices of dependent queries
    priority: str = "medium"  # high, medium, low
    keywords: List[str] = field(default_factory=list)
    entity_focus: Optional[str] = None


@dataclass
class EnhancedQuery:
    """Complete enhanced query with all components"""
    original: str
    sub_queries: List[SubQuery]
    expanded_terms: Dict[str, List[str]]  # term -> synonyms
    variations: List[str]
    query_type: QueryType
    complexity_score: float
    entities: List[str]
    key_concepts: List[str]
    reasoning_path: Optional[str] = None


class QueryEnhancer:
    """
    Decomposes complex queries, expands with context, and generates variations
    for improved retrieval performance.
    """
    
    def __init__(self, bedrock_client=None, use_local_models: bool = True, openai_client=None):
        """
        Initialize QueryEnhancer.
        
        Args:
            bedrock_client: AWS Bedrock client for LLM-based decomposition
            use_local_models: Whether to use local NLP models for expansion
            openai_client: OpenAI client for using Claude Haiku
        """
        self.bedrock_client = bedrock_client
        self.openai_client = openai_client
        self.use_local_models = use_local_models and SPACY_AVAILABLE and SENTENCE_TRANSFORMER_AVAILABLE
        
        if self.use_local_models:
            try:
                # Load spaCy model for NER and POS tagging
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for query analysis")
            except:
                logger.warning("spaCy model not available, will use Claude Haiku instead")
                self.use_local_models = False
                self.nlp = None
            
            if SENTENCE_TRANSFORMER_AVAILABLE:
                # Load sentence transformer for semantic similarity
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            else:
                self.sentence_model = None
        else:
            self.nlp = None
            self.sentence_model = None
            logger.info("Using Claude Haiku for query enhancement")
        
        # Synonym dictionary for common technical terms
        self.synonym_dict = {
            'auth': ['authentication', 'authorization', 'login', 'signin'],
            'api': ['interface', 'endpoint', 'service', 'REST API'],
            'db': ['database', 'datastore', 'storage', 'repository'],
            'config': ['configuration', 'settings', 'parameters', 'options'],
            'error': ['exception', 'failure', 'issue', 'problem', 'bug'],
            'user': ['client', 'customer', 'account', 'member'],
            'create': ['add', 'insert', 'generate', 'make', 'build'],
            'delete': ['remove', 'drop', 'destroy', 'purge', 'clear'],
            'update': ['modify', 'change', 'edit', 'alter', 'patch'],
            'get': ['fetch', 'retrieve', 'read', 'find', 'query'],
            'performance': ['speed', 'efficiency', 'optimization', 'latency'],
            'security': ['protection', 'safety', 'vulnerability', 'threat'],
            'deploy': ['release', 'publish', 'ship', 'rollout', 'launch'],
            'test': ['validate', 'verify', 'check', 'QA', 'testing'],
            'log': ['logging', 'audit', 'trace', 'record', 'track'],
        }
        
        logger.info("Query Enhancer initialized")
    
    async def enhance_query(
        self,
        query: str,
        context: Optional[str] = None,
        max_sub_queries: int = 5
    ) -> EnhancedQuery:
        """
        Enhance a query through decomposition, expansion, and variation.
        
        Args:
            query: Original query text
            context: Optional context for better understanding
            max_sub_queries: Maximum number of sub-queries to generate
            
        Returns:
            Enhanced query with all components
        """
        # Analyze query complexity and type
        query_type, complexity = self._analyze_query(query)
        
        # Extract entities and key concepts
        entities, concepts = self._extract_entities_and_concepts(query)
        
        # Decompose if complex
        sub_queries = []
        if complexity > 0.5 or self._needs_decomposition(query):
            sub_queries = await self._decompose_query(query, context, max_sub_queries)
        else:
            # Create single sub-query for simple queries
            sub_queries = [SubQuery(
                question=query,
                type=query_type,
                dependencies=[],
                priority="high",
                keywords=concepts,
                entity_focus=entities[0] if entities else None
            )]
        
        # Expand terms with synonyms
        expanded_terms = self._expand_terms(query, concepts)
        
        # Generate query variations
        variations = self._generate_variations(query, expanded_terms)
        
        # Determine reasoning path if multi-hop
        reasoning_path = None
        if query_type == QueryType.MULTI_HOP or len(sub_queries) > 2:
            reasoning_path = self._determine_reasoning_path(sub_queries)
        
        return EnhancedQuery(
            original=query,
            sub_queries=sub_queries,
            expanded_terms=expanded_terms,
            variations=variations,
            query_type=query_type,
            complexity_score=complexity,
            entities=entities,
            key_concepts=concepts,
            reasoning_path=reasoning_path
        )
    
    def _analyze_query(self, query: str) -> Tuple[QueryType, float]:
        """Analyze query to determine type and complexity"""
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Patterns for different query types
        factual_patterns = ['what is', 'who is', 'when did', 'where is', 'define']
        analytical_patterns = ['how does', 'why does', 'explain', 'analyze', 'impact']
        comparative_patterns = ['compare', 'difference', 'better', 'worse', 'versus', 'vs']
        navigational_patterns = ['find', 'locate', 'show me', 'where can i', 'documentation']
        multi_hop_patterns = ['and', 'then', 'affect', 'cause', 'lead to', 'result in']
        
        # Score each type
        scores = {
            QueryType.FACTUAL: sum(1 for p in factual_patterns if p in query_lower),
            QueryType.ANALYTICAL: sum(1 for p in analytical_patterns if p in query_lower),
            QueryType.COMPARATIVE: sum(1 for p in comparative_patterns if p in query_lower),
            QueryType.NAVIGATIONAL: sum(1 for p in navigational_patterns if p in query_lower),
            QueryType.MULTI_HOP: sum(1 for p in multi_hop_patterns if p in query_lower),
        }
        
        # Add exploratory score based on open-endedness
        if query.endswith('?') and word_count > 10:
            scores[QueryType.EXPLORATORY] = 2
        
        # Determine primary type
        query_type = max(scores.items(), key=lambda x: x[1])[0] if max(scores.values()) > 0 else QueryType.FACTUAL
        
        # Calculate complexity (0-1 scale)
        complexity_factors = [
            word_count / 30,  # Length factor
            len([c for c in query if c in ',;']) / 5,  # Clause complexity
            sum(scores.values()) / 10,  # Mixed type complexity
            1 if 'and' in query_lower and 'or' in query_lower else 0,  # Boolean complexity
        ]
        
        complexity = min(1.0, sum(complexity_factors) / len(complexity_factors))
        
        return query_type, complexity
    
    def _needs_decomposition(self, query: str) -> bool:
        """Determine if query needs decomposition"""
        indicators = [
            ' and ' in query.lower(),
            ' then ' in query.lower(),
            ' before ' in query.lower(),
            ' after ' in query.lower(),
            '?' in query and query.count('?') > 1,
            len(query.split(',')) > 2,
            len(query.split()) > 15,
        ]
        
        return sum(indicators) >= 2
    
    def _extract_entities_and_concepts(self, query: str) -> Tuple[List[str], List[str]]:
        """Extract named entities and key concepts from query"""
        entities = []
        concepts = []
        
        if self.use_local_models and self.nlp:
            doc = self.nlp(query)
            
            # Extract named entities
            entities = list(set([ent.text for ent in doc.ents]))
            
            # Extract key concepts (nouns and noun phrases)
            for token in doc:
                if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
                    concepts.append(token.text.lower())
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Limit phrase length
                    concepts.append(chunk.text.lower())
        elif self.bedrock_client:
            # Use Bedrock Claude 3.5 Haiku for entity and concept extraction
            try:
                prompt = f"""Extract entities and concepts from the following query for a RAG system.

Query: "{query}"

Return ONLY a valid JSON object with this exact structure:
{{
  "entities": ["entity1", "entity2"],  // Named entities like people, places, organizations, systems
  "concepts": ["concept1", "concept2"]  // Key topics, technical terms, domain concepts
}}

Rules:
- Extract up to 5 entities and 10 concepts
- Entities should be proper nouns when possible
- Concepts should be meaningful keywords for search
- Keep items concise (1-3 words each)
- Return valid JSON only, no other text"""

                response = self.bedrock_client.invoke_model(
                    modelId='us.anthropic.claude-3-5-haiku-20241022-v1:0',
                    contentType='application/json',
                    accept='application/json',
                    body=json.dumps({
                        'messages': [{'role': 'user', 'content': prompt}],
                        'anthropic_version': 'bedrock-2023-05-31',
                        'max_tokens': 200,
                        'temperature': 0.3
                    })
                )
                
                result = json.loads(response['body'].read())
                content = result.get('content', [{}])[0].get('text', '{}')
                extracted = json.loads(content)
                entities = extracted.get('entities', [])
                concepts = extracted.get('concepts', [])
            except Exception as e:
                logger.warning(f"Bedrock extraction failed: {e}, using fallback")
                # Fallback to simple extraction
                words = query.lower().split()
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                             'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were',
                             'what', 'when', 'where', 'who', 'why', 'how', 'which', 'does', 'do'}
                concepts = [w for w in words if w not in stop_words and len(w) > 2]
        else:
            # Fallback to simple keyword extraction
            words = query.lower().split()
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                         'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were',
                         'what', 'when', 'where', 'who', 'why', 'how', 'which', 'does', 'do'}
            
            concepts = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Deduplicate while preserving order
        seen = set()
        unique_concepts = []
        for c in concepts:
            if c not in seen:
                seen.add(c)
                unique_concepts.append(c)
        
        return entities, unique_concepts[:10]  # Limit to top 10 concepts
    
    async def _decompose_query(
        self,
        query: str,
        context: Optional[str],
        max_sub_queries: int
    ) -> List[SubQuery]:
        """Decompose complex query into sub-queries"""
        
        if self.bedrock_client:
            # Use LLM for intelligent decomposition
            return await self._llm_decompose(query, context, max_sub_queries)
        else:
            # Use rule-based decomposition
            return self._rule_based_decompose(query, max_sub_queries)
    
    async def _llm_decompose(
        self,
        query: str,
        context: Optional[str],
        max_sub_queries: int
    ) -> List[SubQuery]:
        """Use LLM to decompose query"""
        
        prompt = f"""Decompose this complex query into simpler sub-questions.
Each sub-question should be independently answerable.
Together, they should fully address the original query.

Complex Query: {query}
{"Context: " + context if context else ""}

Provide {max_sub_queries} or fewer sub-questions in JSON format:
{{
  "sub_questions": [
    {{
      "question": "sub-question text",
      "type": "factual|analytical|comparative|exploratory",
      "dependencies": [indices of questions this depends on],
      "priority": "high|medium|low"
    }}
  ]
}}"""
        
        try:
            # Call Bedrock with Claude 3.5 Haiku for decomposition
            response = await self.bedrock_client.invoke_model(
                modelId='us.anthropic.claude-3-5-haiku-20241022-v1:0',
                body={
                    'prompt': prompt,
                    'max_tokens': 500,
                    'temperature': 0.3,
                    'anthropic_version': 'bedrock-2023-05-31'
                }
            )
            
            # Parse response
            result = json.loads(response)
            sub_queries = []
            
            for sq in result.get('sub_questions', []):
                sub_query = SubQuery(
                    question=sq['question'],
                    type=QueryType(sq.get('type', 'factual')),
                    dependencies=sq.get('dependencies', []),
                    priority=sq.get('priority', 'medium')
                )
                
                # Extract keywords for each sub-query
                _, concepts = self._extract_entities_and_concepts(sq['question'])
                sub_query.keywords = concepts
                
                sub_queries.append(sub_query)
            
            return sub_queries
            
        except Exception as e:
            logger.warning(f"LLM decomposition failed: {e}, falling back to rule-based")
            return self._rule_based_decompose(query, max_sub_queries)
    
    def _rule_based_decompose(self, query: str, max_sub_queries: int) -> List[SubQuery]:
        """Rule-based query decomposition"""
        sub_queries = []
        
        # Split by conjunctions and punctuation
        parts = re.split(r'\s+(?:and|then|but|or)\s+|[,;?]', query)
        parts = [p.strip() for p in parts if p.strip()]
        
        for i, part in enumerate(parts[:max_sub_queries]):
            # Ensure part is a complete question
            if not part.endswith('?'):
                # Try to infer question format
                if part.startswith(('what', 'when', 'where', 'who', 'why', 'how')):
                    part += '?'
                else:
                    part = f"What about {part}?"
            
            # Analyze each part
            query_type, _ = self._analyze_query(part)
            _, concepts = self._extract_entities_and_concepts(part)
            
            # Determine dependencies (simple heuristic: later parts may depend on earlier)
            dependencies = []
            if i > 0 and any(word in part.lower() for word in ['it', 'this', 'that', 'these', 'those']):
                dependencies.append(i - 1)
            
            sub_queries.append(SubQuery(
                question=part,
                type=query_type,
                dependencies=dependencies,
                priority="high" if i == 0 else "medium",
                keywords=concepts
            ))
        
        return sub_queries
    
    def _expand_terms(self, query: str, concepts: List[str]) -> Dict[str, List[str]]:
        """Expand terms with synonyms and related words"""
        expanded = {}
        query_lower = query.lower()
        
        # Expand from predefined synonyms
        for term, synonyms in self.synonym_dict.items():
            if term in query_lower or term in concepts:
                expanded[term] = synonyms
        
        # Expand concepts found in query
        for concept in concepts:
            if concept not in expanded:
                # Look for partial matches in synonym dict
                for term, synonyms in self.synonym_dict.items():
                    if concept in term or term in concept:
                        expanded[concept] = synonyms
                        break
        
        # Add morphological variations
        if self.use_local_models and self.nlp:
            doc = self.nlp(query)
            for token in doc:
                if token.pos_ == 'VERB':
                    # Add verb variations
                    base = token.lemma_
                    variations = [base, base + 's', base + 'ed', base + 'ing']
                    if token.text not in expanded:
                        expanded[token.text] = [v for v in variations if v != token.text]
        
        return expanded
    
    def _generate_variations(self, query: str, expanded_terms: Dict[str, List[str]]) -> List[str]:
        """Generate query variations using expanded terms"""
        variations = [query]  # Original query is first variation
        
        # Generate variations by substituting expanded terms
        for term, synonyms in expanded_terms.items():
            for synonym in synonyms[:2]:  # Limit to avoid explosion
                # Create variation by replacing term
                variation = query.lower().replace(term.lower(), synonym)
                if variation != query.lower() and variation not in variations:
                    variations.append(variation)
        
        # Generate question format variations
        if query.endswith('?'):
            base = query[:-1]
            
            # Different question formats
            formats = [
                f"Can you explain {base}?",
                f"Tell me about {base}",
                f"What do you know about {base}?",
                f"Show me information on {base}"
            ]
            
            for fmt in formats:
                if fmt.lower() not in [v.lower() for v in variations]:
                    variations.append(fmt)
        
        # Limit total variations
        return variations[:7]
    
    def _determine_reasoning_path(self, sub_queries: List[SubQuery]) -> str:
        """Determine the reasoning path for multi-hop queries"""
        if not sub_queries:
            return None
        
        # Build dependency graph
        path_description = []
        
        # Find root queries (no dependencies)
        roots = [i for i, sq in enumerate(sub_queries) if not sq.dependencies]
        
        if roots:
            path_description.append(f"Start with: {', '.join([sub_queries[i].question for i in roots])}")
        
        # Follow dependency chain
        for i, sq in enumerate(sub_queries):
            if sq.dependencies:
                deps = [sub_queries[d].question for d in sq.dependencies if d < len(sub_queries)]
                if deps:
                    path_description.append(f"Then use results from ({', '.join(deps)}) to answer: {sq.question}")
        
        return " → ".join(path_description) if path_description else None
    
    def combine_sub_query_results(
        self,
        sub_queries: List[SubQuery],
        sub_results: List[List[Any]]
    ) -> str:
        """Combine results from sub-queries into coherent answer"""
        if not sub_queries or not sub_results:
            return ""
        
        # Build combined context respecting dependencies
        combined_context = []
        
        for i, (sq, results) in enumerate(zip(sub_queries, sub_results)):
            if results:
                # Add sub-query as section header
                combined_context.append(f"\n## {sq.question}\n")
                
                # Add results
                if isinstance(results, list):
                    for r in results[:3]:  # Limit per sub-query
                        if hasattr(r, 'content'):
                            combined_context.append(r.content)
                        else:
                            combined_context.append(str(r))
                else:
                    combined_context.append(str(results))
        
        return "\n".join(combined_context)
    
    def get_query_embedding_weights(self, enhanced_query: EnhancedQuery) -> Dict[str, float]:
        """Get embedding weights for different query components"""
        weights = {
            'original': 1.0,
            'sub_queries': 0.8,
            'variations': 0.6,
            'expanded': 0.4
        }
        
        # Adjust weights based on query type
        if enhanced_query.query_type == QueryType.MULTI_HOP:
            weights['sub_queries'] = 1.0  # Sub-queries are more important
        elif enhanced_query.query_type == QueryType.NAVIGATIONAL:
            weights['expanded'] = 0.2  # Less emphasis on expansion
        
        return weights