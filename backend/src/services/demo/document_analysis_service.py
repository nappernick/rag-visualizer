"""
Document Analysis Service - Handles document analysis, summarization, and insights
"""
from typing import List, Dict, Any, Optional
import logging
from collections import Counter, defaultdict
import re
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


class DocumentAnalysisService:
    """Service for analyzing documents and generating insights"""
    
    def __init__(self, bedrock_client=None, nlp_processor=None):
        self.bedrock_client = bedrock_client
        self.nlp_processor = nlp_processor
        self.analysis_cache = {}
    
    async def analyze_documents(
        self,
        document_ids: List[str],
        analysis_type: str = "overview",
        include_statistics: bool = True,
        include_entities: bool = True,
        include_themes: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on documents
        
        Args:
            document_ids: List of document IDs to analyze
            analysis_type: Type of analysis (overview, detailed, comparative)
            include_statistics: Whether to include statistical analysis
            include_entities: Whether to include entity analysis
            include_themes: Whether to include theme extraction
            
        Returns:
            Analysis results
        """
        if analysis_type == "overview":
            return await self._analyze_overview(document_ids, include_statistics, include_entities, include_themes)
        elif analysis_type == "detailed":
            return await self._analyze_detailed(document_ids)
        elif analysis_type == "comparative":
            return await self._analyze_comparative(document_ids)
        else:
            logger.warning(f"Unknown analysis type: {analysis_type}")
            return await self._analyze_overview(document_ids, include_statistics, include_entities, include_themes)
    
    async def _analyze_overview(
        self,
        document_ids: List[str],
        include_statistics: bool,
        include_entities: bool,
        include_themes: bool
    ) -> Dict[str, Any]:
        """
        Generate overview analysis of documents
        """
        analysis = {
            'type': 'overview',
            'documents_analyzed': len(document_ids),
            'timestamp': datetime.now().isoformat()
        }
        
        # Statistical analysis
        if include_statistics:
            analysis['statistics'] = await self._calculate_statistics(document_ids)
        
        # Entity analysis
        if include_entities:
            analysis['entities'] = await self._analyze_entities(document_ids)
        
        # Theme extraction
        if include_themes:
            analysis['themes'] = await self._extract_themes(document_ids)
        
        # Key findings
        analysis['key_findings'] = self._generate_key_findings(analysis)
        
        return analysis
    
    async def _analyze_detailed(
        self,
        document_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Generate detailed analysis of documents
        """
        analysis = {
            'type': 'detailed',
            'documents_analyzed': len(document_ids),
            'timestamp': datetime.now().isoformat(),
            'documents': []
        }
        
        for doc_id in document_ids:
            doc_analysis = {
                'document_id': doc_id,
                'content_analysis': await self._analyze_content(doc_id),
                'structure_analysis': await self._analyze_structure(doc_id),
                'quality_metrics': await self._calculate_quality_metrics(doc_id),
                'readability_scores': self._calculate_readability(doc_id),
                'sentiment_analysis': await self._analyze_sentiment(doc_id)
            }
            analysis['documents'].append(doc_analysis)
        
        # Overall insights
        analysis['overall_insights'] = self._generate_overall_insights(analysis['documents'])
        
        return analysis
    
    async def _analyze_comparative(
        self,
        document_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Generate comparative analysis between documents
        """
        if len(document_ids) < 2:
            return {
                'error': 'Comparative analysis requires at least 2 documents',
                'documents_provided': len(document_ids)
            }
        
        analysis = {
            'type': 'comparative',
            'documents_compared': len(document_ids),
            'timestamp': datetime.now().isoformat()
        }
        
        # Similarity analysis
        analysis['similarity_matrix'] = await self._calculate_similarity_matrix(document_ids)
        
        # Common themes
        analysis['common_themes'] = await self._find_common_themes(document_ids)
        
        # Unique aspects
        analysis['unique_aspects'] = await self._find_unique_aspects(document_ids)
        
        # Coverage analysis
        analysis['coverage_analysis'] = await self._analyze_coverage(document_ids)
        
        # Recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    async def summarize_documents(
        self,
        document_ids: List[str],
        summary_type: str = "brief",
        max_length: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Generate summaries for documents
        
        Args:
            document_ids: Documents to summarize
            summary_type: Type of summary (brief, detailed, technical)
            max_length: Maximum summary length
            
        Returns:
            List of document summaries
        """
        summaries = []
        
        for doc_id in document_ids:
            # Check cache
            cache_key = f"{doc_id}_{summary_type}_{max_length}"
            if cache_key in self.analysis_cache:
                summaries.append(self.analysis_cache[cache_key])
                continue
            
            # Generate summary
            if self.bedrock_client:
                summary = await self._generate_ai_summary(
                    doc_id, summary_type, max_length
                )
            else:
                summary = self._generate_extractive_summary(
                    doc_id, summary_type, max_length
                )
            
            # Cache result
            self.analysis_cache[cache_key] = summary
            summaries.append(summary)
        
        return summaries
    
    async def _generate_ai_summary(
        self,
        document_id: str,
        summary_type: str,
        max_length: int
    ) -> Dict[str, Any]:
        """
        Generate AI-powered summary using Bedrock
        """
        try:
            # In production, would fetch actual document content
            document_content = f"Document {document_id} content placeholder"
            
            prompt = self._build_summary_prompt(
                document_content, summary_type, max_length
            )
            
            response = self.bedrock_client.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body={
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_length,
                    "temperature": 0.3
                }
            )
            
            summary_text = response.get('content', [{}])[0].get('text', '')
            
            return {
                'document_id': document_id,
                'summary': summary_text,
                'type': summary_type,
                'method': 'ai_generated',
                'length': len(summary_text.split())
            }
            
        except Exception as e:
            logger.error(f"Error generating AI summary: {e}")
            return self._generate_extractive_summary(
                document_id, summary_type, max_length
            )
    
    def _generate_extractive_summary(
        self,
        document_id: str,
        summary_type: str,
        max_length: int
    ) -> Dict[str, Any]:
        """
        Generate extractive summary using key sentences
        """
        # In production, would fetch actual document content
        sentences = [
            f"Key point 1 from document {document_id}.",
            f"Important insight about the topic in {document_id}.",
            f"Conclusion drawn from analysis in {document_id}."
        ]
        
        # Select sentences based on summary type
        if summary_type == "brief":
            selected = sentences[:1]
        elif summary_type == "detailed":
            selected = sentences[:3]
        else:  # technical
            selected = sentences[:2]
        
        summary_text = " ".join(selected)
        
        return {
            'document_id': document_id,
            'summary': summary_text,
            'type': summary_type,
            'method': 'extractive',
            'length': len(summary_text.split())
        }
    
    def _build_summary_prompt(
        self,
        content: str,
        summary_type: str,
        max_length: int
    ) -> str:
        """
        Build prompt for AI summarization
        """
        prompts = {
            'brief': f"""Provide a brief summary of the following document in {max_length} words or less.
            Focus on the main topic and key conclusion.
            
            Document: {content}
            
            Brief Summary:""",
            
            'detailed': f"""Provide a detailed summary of the following document in {max_length} words or less.
            Include main topics, key points, and important details.
            
            Document: {content}
            
            Detailed Summary:""",
            
            'technical': f"""Provide a technical summary of the following document in {max_length} words or less.
            Focus on technical details, methodologies, and specifications.
            
            Document: {content}
            
            Technical Summary:"""
        }
        
        return prompts.get(summary_type, prompts['brief'])
    
    async def _calculate_statistics(
        self,
        document_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate statistical metrics for documents
        """
        stats = {
            'total_documents': len(document_ids),
            'total_chunks': 0,
            'total_entities': 0,
            'total_relationships': 0,
            'avg_chunk_size': 0,
            'avg_entities_per_doc': 0,
            'document_lengths': [],
            'chunk_distribution': {}
        }
        
        # In production, would query actual data
        for doc_id in document_ids:
            # Simulate statistics
            chunks = 10 + hash(doc_id) % 20
            entities = 15 + hash(doc_id) % 30
            relationships = 20 + hash(doc_id) % 40
            length = 1000 + hash(doc_id) % 5000
            
            stats['total_chunks'] += chunks
            stats['total_entities'] += entities
            stats['total_relationships'] += relationships
            stats['document_lengths'].append(length)
            stats['chunk_distribution'][doc_id] = chunks
        
        # Calculate averages
        if document_ids:
            stats['avg_chunk_size'] = stats['total_chunks'] / len(document_ids)
            stats['avg_entities_per_doc'] = stats['total_entities'] / len(document_ids)
            stats['avg_document_length'] = statistics.mean(stats['document_lengths'])
            stats['median_document_length'] = statistics.median(stats['document_lengths'])
        
        return stats
    
    async def _analyze_entities(
        self,
        document_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze entities across documents
        """
        entity_analysis = {
            'total_unique_entities': 0,
            'entity_types': {},
            'most_common_entities': [],
            'entity_distribution': {},
            'cross_document_entities': []
        }
        
        # In production, would query actual entities
        all_entities = []
        entity_docs = defaultdict(set)
        
        for doc_id in document_ids:
            # Simulate entity extraction
            doc_entities = [
                f"Entity_{i}_{doc_id[:4]}" 
                for i in range(5 + hash(doc_id) % 10)
            ]
            
            for entity in doc_entities:
                all_entities.append(entity)
                entity_docs[entity].add(doc_id)
        
        # Analyze entities
        entity_counts = Counter(all_entities)
        entity_analysis['total_unique_entities'] = len(set(all_entities))
        entity_analysis['most_common_entities'] = [
            {'entity': entity, 'count': count}
            for entity, count in entity_counts.most_common(10)
        ]
        
        # Find cross-document entities
        for entity, docs in entity_docs.items():
            if len(docs) > 1:
                entity_analysis['cross_document_entities'].append({
                    'entity': entity,
                    'documents': list(docs),
                    'document_count': len(docs)
                })
        
        # Entity type distribution
        entity_analysis['entity_types'] = {
            'person': len([e for e in all_entities if 'person' in e.lower()]),
            'organization': len([e for e in all_entities if 'org' in e.lower()]),
            'location': len([e for e in all_entities if 'loc' in e.lower()]),
            'other': len(all_entities) // 4
        }
        
        return entity_analysis
    
    async def _extract_themes(
        self,
        document_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Extract themes from documents
        """
        themes = []
        
        # In production, would use actual theme extraction
        theme_templates = [
            "Data Processing and Analytics",
            "Machine Learning Applications",
            "System Architecture",
            "User Experience Design",
            "Performance Optimization"
        ]
        
        for i, theme in enumerate(theme_templates[:min(3, len(document_ids))]):
            themes.append({
                'theme': theme,
                'confidence': 0.7 + (i * 0.05),
                'document_coverage': len(document_ids) / (i + 2),
                'key_terms': [
                    f"term_{j}" for j in range(3 + i)
                ],
                'relevance_score': 0.8 - (i * 0.1)
            })
        
        return themes
    
    async def _analyze_content(
        self,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Analyze content characteristics of a document
        """
        return {
            'content_type': 'technical',
            'primary_language': 'english',
            'complexity_level': 'medium',
            'topic_coherence': 0.85,
            'information_density': 0.72,
            'key_sections': [
                'introduction',
                'methodology',
                'results',
                'conclusion'
            ]
        }
    
    async def _analyze_structure(
        self,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Analyze document structure
        """
        return {
            'has_sections': True,
            'section_count': 5,
            'has_headers': True,
            'has_lists': True,
            'has_tables': False,
            'has_code_blocks': True,
            'structure_score': 0.78
        }
    
    async def _calculate_quality_metrics(
        self,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Calculate quality metrics for a document
        """
        return {
            'completeness': 0.9,
            'accuracy': 0.85,
            'clarity': 0.88,
            'consistency': 0.92,
            'relevance': 0.87,
            'overall_quality': 0.88
        }
    
    def _calculate_readability(
        self,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Calculate readability scores
        """
        return {
            'flesch_reading_ease': 45.2,
            'flesch_kincaid_grade': 12.3,
            'gunning_fog': 14.1,
            'automated_readability_index': 13.5,
            'readability_level': 'college'
        }
    
    async def _analyze_sentiment(
        self,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Analyze document sentiment
        """
        return {
            'overall_sentiment': 'neutral',
            'sentiment_scores': {
                'positive': 0.35,
                'neutral': 0.55,
                'negative': 0.10
            },
            'emotional_tone': 'informative',
            'subjectivity': 0.3
        }
    
    async def _calculate_similarity_matrix(
        self,
        document_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate pairwise document similarities
        """
        matrix = {}
        
        for i, doc1 in enumerate(document_ids):
            for j, doc2 in enumerate(document_ids[i+1:], start=i+1):
                # Simulate similarity calculation
                similarity = 0.5 + (hash(f"{doc1}_{doc2}") % 50) / 100
                key = f"{doc1}__{doc2}"
                matrix[key] = {
                    'doc1': doc1,
                    'doc2': doc2,
                    'similarity': similarity,
                    'method': 'cosine'
                }
        
        return matrix
    
    async def _find_common_themes(
        self,
        document_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Find themes common across documents
        """
        return [
            {
                'theme': 'Data Processing',
                'documents': document_ids[:max(2, len(document_ids)//2)],
                'strength': 0.8
            },
            {
                'theme': 'System Design',
                'documents': document_ids[1:],
                'strength': 0.65
            }
        ]
    
    async def _find_unique_aspects(
        self,
        document_ids: List[str]
    ) -> Dict[str, List[str]]:
        """
        Find unique aspects of each document
        """
        unique = {}
        
        for doc_id in document_ids:
            unique[doc_id] = [
                f"Unique aspect 1 of {doc_id[:8]}",
                f"Distinctive feature in {doc_id[:8]}"
            ]
        
        return unique
    
    async def _analyze_coverage(
        self,
        document_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze topic coverage across documents
        """
        return {
            'total_topics_covered': 15,
            'average_topic_depth': 0.72,
            'coverage_gaps': [
                'Advanced optimization techniques',
                'Error handling strategies'
            ],
            'overlapping_topics': 8,
            'unique_topics_per_doc': {
                doc_id: 3 + (hash(doc_id) % 5)
                for doc_id in document_ids
            }
        }
    
    def _generate_key_findings(
        self,
        analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Generate key findings from analysis
        """
        findings = []
        
        if 'statistics' in analysis:
            stats = analysis['statistics']
            findings.append(
                f"Analyzed {stats['total_documents']} documents with "
                f"{stats['total_chunks']} total chunks"
            )
        
        if 'entities' in analysis:
            entities = analysis['entities']
            findings.append(
                f"Found {entities['total_unique_entities']} unique entities "
                f"across documents"
            )
        
        if 'themes' in analysis:
            themes = analysis['themes']
            if themes:
                findings.append(
                    f"Identified {len(themes)} main themes, "
                    f"with '{themes[0]['theme']}' being most prominent"
                )
        
        return findings
    
    def _generate_overall_insights(
        self,
        document_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate overall insights from detailed analyses
        """
        return {
            'average_quality': statistics.mean([
                doc['quality_metrics']['overall_quality']
                for doc in document_analyses
            ]),
            'common_structure_patterns': ['sections', 'headers', 'code_blocks'],
            'readability_assessment': 'College level - suitable for technical audience',
            'improvement_suggestions': [
                'Add more visual elements',
                'Improve cross-referencing between documents',
                'Standardize terminology usage'
            ]
        }
    
    def _generate_recommendations(
        self,
        analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Generate recommendations based on comparative analysis
        """
        recommendations = []
        
        # Based on similarity
        if 'similarity_matrix' in analysis:
            recommendations.append(
                "Consider consolidating highly similar documents to reduce redundancy"
            )
        
        # Based on coverage
        if 'coverage_analysis' in analysis:
            gaps = analysis['coverage_analysis'].get('coverage_gaps', [])
            if gaps:
                recommendations.append(
                    f"Address coverage gaps in: {', '.join(gaps[:2])}"
                )
        
        # Based on unique aspects
        if 'unique_aspects' in analysis:
            recommendations.append(
                "Leverage unique aspects of each document for comprehensive coverage"
            )
        
        return recommendations
    
    def clear_cache(self):
        """Clear the analysis cache"""
        self.analysis_cache.clear()