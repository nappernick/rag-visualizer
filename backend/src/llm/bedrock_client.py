"""
AWS Bedrock Claude Client for RAG generation
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

class BedrockClaudeClient:
    """Client for AWS Bedrock Claude 3.5 Sonnet"""
    
    def __init__(self):
        """Initialize Bedrock client"""
        # AWS credentials from environment
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION", "us-east-2")
        
        # Claude 3.5 Sonnet model ID with US cross-region inference
        # Using the inference profile for better availability
        self.model_id = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")
        
        # Initialize Bedrock runtime client
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.aws_region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            config=Config(
                region_name=self.aws_region,
                retries={'max_attempts': 3, 'mode': 'adaptive'}
            )
        )
        
        logger.info(f"Initialized Bedrock client with model: {self.model_id}")
    
    def generate(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response using Claude 3.5 Sonnet
        
        Args:
            query: User query
            context: List of context chunks from retrieval
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        try:
            # Prepare context
            context_text = "\n\n".join(context)
            
            # Default system prompt for RAG
            if not system_prompt:
                system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
                Use the context to provide accurate and relevant answers.
                If the context doesn't contain enough information to answer the question, say so.
                Be concise but thorough in your responses."""
            
            # Prepare the prompt
            user_message = f"""Context:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above."""
            
            # Prepare the request body for Claude 3.5
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            }
            
            # Invoke the model
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Extract the generated text
            if 'content' in response_body and len(response_body['content']) > 0:
                generated_text = response_body['content'][0].get('text', '')
            else:
                generated_text = "Unable to generate response."
            
            logger.info(f"Generated response of length: {len(generated_text)}")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating with Bedrock Claude: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_with_tools(
        self,
        query: str,
        context: List[str],
        tools: List[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate response with tool use capability
        
        Args:
            query: User query
            context: List of context chunks
            tools: List of tool definitions
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            
        Returns:
            Dictionary with response and any tool calls
        """
        try:
            # Prepare context
            context_text = "\n\n".join(context)
            
            # Prepare the user message
            user_message = f"""Context:
{context_text}

Question: {query}"""
            
            # Prepare request with tools if provided
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            }
            
            if system_prompt:
                request_body["system"] = system_prompt
            
            if tools:
                request_body["tools"] = tools
                request_body["tool_choice"] = {"type": "auto"}
            
            # Invoke the model
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            return response_body
            
        except Exception as e:
            logger.error(f"Error with tool use: {e}")
            return {"error": str(e)}
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings using Bedrock (if available)
        Note: Claude doesn't directly provide embeddings, 
        so this would need a different model like Titan Embeddings
        """
        # For now, return None as Claude doesn't do embeddings
        # You would use Amazon Titan Embeddings or another model
        return None