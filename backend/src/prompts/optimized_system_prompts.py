"""SPRIG-based optimized system prompts for enhanced RAG performance.

Implements findings from the SPRIG paper showing 15-25% performance improvements
through emotional stimuli, structured formatting, and position bias optimization.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PromptConfig:
    """Configuration for prompt optimization."""
    use_emotional_stimuli: bool = True
    use_xml_structure: bool = True
    optimize_position_bias: bool = True
    include_reasoning_triggers: bool = True


class OptimizedSystemPrompts:
    """SPRIG-based prompt optimization with emotional stimuli and structured formatting."""
    
    def __init__(self, config: Optional[PromptConfig] = None):
        self.config = config or PromptConfig()
    
    def get_rag_query_prompt(self, 
                             query: str, 
                             primary_context: str,
                             supporting_context: str = "",
                             metadata: Optional[Dict] = None) -> str:
        """Generate optimized RAG query prompt with position bias optimization.
        
        Position bias findings:
        - Beginning: 95% recall
        - Middle: 65% recall  
        - End: 90% recall
        """
        
        emotional_preamble = ""
        if self.config.use_emotional_stimuli:
            emotional_preamble = """
    Your analysis directly impacts critical decision-making processes.
    Excellence in your response is essential for project success.
    Take pride in providing accurate, comprehensive information.
    """
        
        reasoning_trigger = ""
        if self.config.include_reasoning_triggers:
            reasoning_trigger = """
    Think step-by-step through the evidence before responding.
    Consider multiple perspectives and potential implications.
    """
        
        if self.config.use_xml_structure:
            prompt = f"""
<system>
You are an expert information retrieval specialist with deep knowledge across domains.
{emotional_preamble}
</system>

<task>
Analyze the provided context and answer the query with precision.
{reasoning_trigger}
</task>

<context position="primary" importance="critical">
{primary_context}
</context>

<query>
{query}
</query>

<context position="supporting" importance="supplementary">
{supporting_context}
</context>

<instructions>
1. Identify key information relevant to the query
2. Synthesize evidence from multiple sources
3. Provide a comprehensive yet concise answer
4. Cite specific sources when making claims
5. Acknowledge any limitations or uncertainties
</instructions>

<output_requirements>
- Be precise and factual
- Support claims with evidence
- Maintain logical flow
- Address the query directly
</output_requirements>
"""
        else:
            # Fallback to standard format
            prompt = f"""
You are an expert information retrieval specialist.
{emotional_preamble}

Context (Primary):
{primary_context}

Query: {query}

Additional Context:
{supporting_context}

Please provide a comprehensive answer based on the context provided.
{reasoning_trigger}
"""
        
        return prompt.strip()
    
    def get_graph_extraction_prompt(self, 
                                   text: str,
                                   extraction_focus: str = "general") -> str:
        """Generate optimized prompt for entity and relationship extraction."""
        
        focus_instructions = {
            "general": "Extract all significant entities and their relationships",
            "technical": "Focus on technical terms, systems, and their interactions",
            "workflow": "Identify process steps, actors, and flow relationships",
            "conceptual": "Extract abstract concepts and their logical connections"
        }
        
        instruction = focus_instructions.get(extraction_focus, focus_instructions["general"])
        
        if self.config.use_xml_structure:
            prompt = f"""
<critical_task>
Extract entities and relationships with the precision of a domain expert.
Your extraction quality determines the system's knowledge representation.
</critical_task>

<text>
{text}
</text>

<extraction_rules>
- Entities: Concrete nouns, proper names, technical terms, key concepts
- Relationships: Verbs connecting entities, implied connections, hierarchies
- Confidence: Rate 0.0-1.0 based on context clarity and evidence strength
- Focus: {instruction}
</extraction_rules>

<output_format>
Provide JSON with structure:
{{
  "entities": [
    {{
      "id": "unique_identifier",
      "text": "entity text",
      "type": "person|organization|concept|system|process|location",
      "confidence": 0.0-1.0,
      "context": "surrounding context"
    }}
  ],
  "relationships": [
    {{
      "source": "entity_id",
      "target": "entity_id",
      "type": "relationship_type",
      "description": "relationship description",
      "confidence": 0.0-1.0
    }}
  ]
}}
</output_format>
"""
        else:
            prompt = f"""
Extract entities and relationships from the following text.
{instruction}

Text: {text}

Provide output as JSON with entities and relationships.
"""
        
        return prompt.strip()
    
    def get_multi_hop_reasoning_prompt(self,
                                      query: str,
                                      reasoning_paths: List[Dict],
                                      context_chunks: List[str]) -> str:
        """Generate prompt for multi-hop reasoning across knowledge graph paths."""
        
        if self.config.use_xml_structure:
            paths_formatted = "\n".join([
                f"<path confidence=\"{p.get('confidence', 0.0)}\">{p.get('description', '')}"</path>"
                for p in reasoning_paths
            ])
            
            context_formatted = "\n".join([
                f"<chunk relevance=\"high\">{chunk}</chunk>"
                for chunk in context_chunks[:3]  # Primary context
            ]) + "\n" + "\n".join([
                f"<chunk relevance=\"medium\">{chunk}</chunk>"
                for chunk in context_chunks[3:6]  # Supporting context
            ])
            
            prompt = f"""
<reasoning_task>
Answer the query by connecting information across multiple sources.
This requires multi-step reasoning to reach the correct conclusion.
</reasoning_task>

<query importance="critical">
{query}
</query>

<reasoning_paths>
{paths_formatted}
</reasoning_paths>

<evidence_chunks>
{context_formatted}
</evidence_chunks>

<reasoning_instructions>
1. Identify the reasoning steps needed to answer the query
2. Connect information from different paths and chunks
3. Build a logical chain from evidence to conclusion
4. Explicitly state each reasoning step
5. Provide confidence in your final answer
</reasoning_instructions>

<output_format>
Provide:
1. Step-by-step reasoning chain
2. Final answer with supporting evidence
3. Confidence score (0.0-1.0)
4. Any assumptions or limitations
</output_format>
"""
        else:
            prompt = f"""
Answer this query using multi-hop reasoning:

Query: {query}

Reasoning paths available:
{reasoning_paths}

Context:
{context_chunks}

Provide step-by-step reasoning and a final answer.
"""
        
        return prompt.strip()
    
    def get_query_decomposition_prompt(self, complex_query: str) -> str:
        """Generate prompt for decomposing complex queries into sub-questions."""
        
        if self.config.use_emotional_stimuli:
            motivation = """Your decomposition will enable more accurate and complete answers.
Breaking this down correctly is crucial for understanding the user's needs."""
        else:
            motivation = ""
        
        if self.config.use_xml_structure:
            prompt = f"""
<task>
Decompose a complex query into simpler, answerable sub-questions.
{motivation}
</task>

<complex_query>
{complex_query}
</complex_query>

<decomposition_rules>
1. Each sub-question should be independently answerable
2. Together, sub-questions should fully address the original query
3. Maintain logical dependencies between questions
4. Identify information types needed (factual, analytical, comparative)
5. Order questions from foundational to derived
</decomposition_rules>

<output_format>
Provide 2-5 sub-questions in JSON:
{{
  "sub_questions": [
    {{
      "question": "sub-question text",
      "type": "factual|analytical|comparative|exploratory",
      "dependencies": ["index of dependent questions"],
      "priority": "high|medium|low"
    }}
  ],
  "reasoning_path": "explanation of how sub-questions connect"
}}
</output_format>
"""
        else:
            prompt = f"""
Decompose this complex query into simpler sub-questions:

Query: {complex_query}

Provide 2-5 focused sub-questions that together answer the main query.
{motivation}
"""
        
        return prompt.strip()
    
    def get_answer_critique_prompt(self,
                                  query: str,
                                  answer: str,
                                  context: str) -> str:
        """Generate prompt for self-critique of generated answers."""
        
        if self.config.use_xml_structure:
            prompt = f"""
<critique_task>
Critically evaluate this answer for accuracy, completeness, and relevance.
Your evaluation will determine if the answer meets quality standards.
</critique_task>

<original_query>
{query}
</original_query>

<generated_answer>
{answer}
</generated_answer>

<available_context>
{context}
</available_context>

<evaluation_criteria>
1. Factual Accuracy: Does the answer align with the provided context?
2. Completeness: Are all aspects of the query addressed?
3. Relevance: Is the information directly relevant to the query?
4. Coherence: Is the answer logically structured and clear?
5. Evidence Support: Are claims properly supported by context?
</evaluation_criteria>

<output_format>
Provide evaluation in JSON:
{{
  "factual_accuracy": 0.0-1.0,
  "completeness": 0.0-1.0,
  "relevance": 0.0-1.0,
  "coherence": 0.0-1.0,
  "evidence_support": 0.0-1.0,
  "overall_score": 0.0-1.0,
  "issues_found": ["list of specific issues"],
  "missing_information": ["what should be added"],
  "improvement_suggestions": ["specific improvements"]
}}
</output_format>
"""
        else:
            prompt = f"""
Critique this answer:

Query: {query}
Answer: {answer}
Context: {context}

Evaluate for accuracy, completeness, relevance, and coherence.
Provide scores and specific improvement suggestions.
"""
        
        return prompt.strip()
    
    def get_diagram_analysis_prompt(self,
                                   diagram_type: str = "workflow",
                                   analysis_focus: str = "structure") -> str:
        """Generate prompt for VLM diagram analysis."""
        
        focus_prompts = {
            "structure": "Identify the overall structure, flow direction, and main components",
            "elements": "Extract all nodes, shapes, and text labels with their positions",
            "relationships": "Map all connections, arrows, and flow relationships",
            "text": "Extract and transcribe all text content accurately"
        }
        
        if self.config.use_xml_structure:
            prompt = f"""
<vision_task>
Analyze this {diagram_type} diagram with expert precision.
Your analysis will be used to build a knowledge representation.
</vision_task>

<analysis_focus>
{focus_prompts.get(analysis_focus, focus_prompts['structure'])}
</analysis_focus>

<extraction_requirements>
1. Overall Layout:
   - Flow direction (left-right, top-bottom, circular, hierarchical)
   - Main sections or swim lanes
   - Groupings or clusters

2. Elements:
   - Shapes (rectangles, circles, diamonds, etc.)
   - Colors and their significance
   - Text labels and annotations
   - Icons or symbols

3. Connections:
   - Arrows and their directions
   - Line styles (solid, dashed, dotted)
   - Connection labels
   - Branching and merging points

4. Semantic Meaning:
   - Process steps and their sequence
   - Decision points and conditions
   - Parallel processes
   - Loops or cycles
</extraction_requirements>

<output_format>
Provide structured JSON with:
{{
  "diagram_type": "identified type",
  "flow_direction": "direction",
  "elements": [{{
    "id": "unique_id",
    "type": "shape_type",
    "text": "label_text",
    "position": {{"x": 0, "y": 0}},
    "properties": {{}}
  }}],
  "connections": [{{
    "source": "element_id",
    "target": "element_id",
    "type": "arrow|line",
    "label": "connection_label"
  }}],
  "confidence": 0.0-1.0
}}
</output_format>
"""
        else:
            prompt = f"""
Analyze this {diagram_type} diagram.
{focus_prompts.get(analysis_focus, focus_prompts['structure'])}

Extract elements, connections, and overall structure.
Provide results in structured JSON format.
"""
        
        return prompt.strip()