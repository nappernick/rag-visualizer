"""
AWS Textract processor for PDF and image OCR
"""
import os
import boto3
import logging
import time
from typing import Optional, Dict, Any
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class TextractProcessor:
    """Process PDFs and images using AWS Textract"""
    
    def __init__(self):
        """Initialize Textract client"""
        self.textract_client = boto3.client(
            'textract',
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-2')
        )
        self.s3_client = boto3.client(
            's3',
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-2')
        )
        self.bucket_name = os.getenv('DIAGRAM_BUCKET', 'rag-visualizer-docs')
        
    def process_document(self, file_content: bytes, file_name: str) -> str:
        """Process a document (PDF or image) and extract text"""
        try:
            # For small documents, use synchronous detection
            if len(file_content) < 5 * 1024 * 1024:  # 5MB
                return self._process_sync(file_content)
            else:
                # For larger documents, upload to S3 and use async
                return self._process_async(file_content, file_name)
                
        except Exception as e:
            logger.error(f"Error processing document with Textract: {e}")
            raise
    
    def _process_sync(self, file_content: bytes) -> str:
        """Process document synchronously for small files"""
        try:
            # Call Textract directly with document bytes
            response = self.textract_client.detect_document_text(
                Document={'Bytes': file_content}
            )
            
            # Extract text from response
            return self._extract_text_from_response(response)
            
        except ClientError as e:
            logger.error(f"Textract sync processing error: {e}")
            raise
    
    def _process_async(self, file_content: bytes, file_name: str) -> str:
        """Process document asynchronously for large files"""
        try:
            # Upload to S3
            s3_key = f"textract-temp/{file_name}"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_content
            )
            
            # Start async Textract job
            response = self.textract_client.start_document_text_detection(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': self.bucket_name,
                        'Name': s3_key
                    }
                }
            )
            
            job_id = response['JobId']
            logger.info(f"Started Textract job: {job_id}")
            
            # Wait for job completion
            text = self._wait_for_job_completion(job_id)
            
            # Clean up S3 object
            try:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            except:
                pass  # Ignore cleanup errors
                
            return text
            
        except ClientError as e:
            logger.error(f"Textract async processing error: {e}")
            raise
    
    def _wait_for_job_completion(self, job_id: str, max_wait: int = 300) -> str:
        """Wait for Textract job to complete and return results"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            response = self.textract_client.get_document_text_detection(JobId=job_id)
            status = response['JobStatus']
            
            if status == 'SUCCEEDED':
                logger.info(f"Textract job {job_id} completed successfully")
                return self._extract_text_from_response(response)
            elif status == 'FAILED':
                raise Exception(f"Textract job failed: {response.get('StatusMessage', 'Unknown error')}")
            
            time.sleep(5)  # Wait 5 seconds before checking again
            
        raise Exception(f"Textract job timed out after {max_wait} seconds")
    
    def _extract_text_from_response(self, response: Dict[str, Any]) -> str:
        """Extract text from Textract response"""
        text_lines = []
        
        blocks = response.get('Blocks', [])
        
        # Process LINE blocks to maintain reading order
        line_blocks = [block for block in blocks if block.get('BlockType') == 'LINE']
        
        # Sort by vertical position, then horizontal
        line_blocks.sort(key=lambda b: (
            b.get('Geometry', {}).get('BoundingBox', {}).get('Top', 0),
            b.get('Geometry', {}).get('BoundingBox', {}).get('Left', 0)
        ))
        
        for block in line_blocks:
            text = block.get('Text', '')
            if text:
                text_lines.append(text)
        
        # Handle pagination for async results
        next_token = response.get('NextToken')
        if next_token:
            next_response = self.textract_client.get_document_text_detection(
                JobId=response.get('JobId'),
                NextToken=next_token
            )
            text_lines.append(self._extract_text_from_response(next_response))
        
        return '\n'.join(text_lines)