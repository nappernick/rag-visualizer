"""
Neo4j Connection Manager with Retry Logic and Health Checks
Based on reference implementation for robust connection handling
"""

import time
import logging
import os
from typing import Optional, Dict, Any, Callable
from functools import wraps
from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

logger = logging.getLogger(__name__)


class Neo4jConnectionManager:
    """Manages Neo4j connections with automatic retry and health checks."""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None, 
        password: Optional[str] = None,
        database: str = "neo4j",
        max_retry_attempts: int = 3,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 10.0
    ):
        """
        Initialize the connection manager.
        
        Args:
            uri: Neo4j connection URI
            user: Username for authentication
            password: Password for authentication
            database: Database name (default: "neo4j")
            max_retry_attempts: Maximum number of retry attempts
            initial_retry_delay: Initial delay between retries in seconds
            max_retry_delay: Maximum delay between retries in seconds
        """
        # Get from environment if not provided
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self.max_retry_attempts = max_retry_attempts
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        
        # Connection configuration with optimized settings
        self.connection_config = {
            "max_connection_pool_size": 50,
            "connection_acquisition_timeout": 30.0,
            "connection_timeout": 10.0,
            "max_transaction_retry_time": 30.0,
            "keep_alive": True
        }
        
        self._driver = None
        self.enabled = False
        
        # Only try to connect if credentials are provided
        if self.password:
            self._connect()
        else:
            logger.info("Neo4j credentials not configured - graph operations will be disabled")
    
    def _connect(self) -> None:
        """Establish connection to Neo4j with retry logic."""
        last_error = None
        retry_delay = self.initial_retry_delay
        
        for attempt in range(self.max_retry_attempts):
            try:
                logger.info(f"Attempting to connect to Neo4j at {self.uri} (attempt {attempt + 1}/{self.max_retry_attempts})")
                
                # Close existing driver if any
                if self._driver:
                    try:
                        self._driver.close()
                    except Exception:
                        pass
                
                # Create new driver with configuration
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=basic_auth(self.user, self.password),
                    **self.connection_config
                )
                
                # Verify connection
                self._driver.verify_connectivity()
                self.enabled = True
                logger.info("✅ Successfully connected to Neo4j")
                return
                
            except ServiceUnavailable as e:
                last_error = e
                logger.warning(f"Neo4j unavailable (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retry_attempts - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, self.max_retry_delay)
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to connect to Neo4j: {e}")
                break
        
        # Connection failed
        self.enabled = False
        logger.warning(f"⚠️ Neo4j connection failed after {self.max_retry_attempts} attempts: {last_error}")
        logger.info("Graph operations will be disabled")
    
    def get_driver(self):
        """Get the Neo4j driver, reconnecting if necessary."""
        if not self._driver and self.password:
            self._connect()
        return self._driver
    
    def health_check(self) -> bool:
        """
        Check if the Neo4j connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            driver = self.get_driver()
            if not driver:
                return False
                
            driver.verify_connectivity()
            
            # Run a simple query to verify database access
            with driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS health")
                value = result.single()["health"]
                return value == 1
                
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False
    
    def ensure_connected(self) -> None:
        """Ensure connection is established, reconnecting if necessary."""
        if not self.health_check() and self.password:
            logger.info("Connection unhealthy, attempting to reconnect...")
            self._connect()
    
    def execute_with_retry(self, query: str, parameters: Optional[Dict] = None) -> Any:
        """
        Execute a query with automatic retry logic.
        
        Args:
            query: Cypher query to execute
            parameters: Query parameters
            
        Returns:
            Query result or None if failed
        """
        if not self.enabled:
            return None
            
        last_error = None
        retry_delay = self.initial_retry_delay
        
        for attempt in range(self.max_retry_attempts):
            try:
                # Ensure connection before operation
                self.ensure_connected()
                
                if not self._driver:
                    return None
                    
                with self._driver.session(database=self.database) as session:
                    result = session.run(query, parameters or {})
                    return list(result)
                    
            except (ServiceUnavailable, SessionExpired, TransientError) as e:
                last_error = e
                logger.warning(f"Transient error in query (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retry_attempts - 1:
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, self.max_retry_delay)
                    # Try to reconnect for next attempt
                    try:
                        self._connect()
                    except Exception:
                        pass
                        
            except Exception as e:
                logger.error(f"Non-retryable error in query: {e}")
                return None
        
        logger.error(f"Query failed after {self.max_retry_attempts} attempts: {last_error}")
        return None
    
    def close(self):
        """Close the Neo4j driver connection."""
        if self._driver:
            try:
                self._driver.close()
            except Exception as e:
                logger.warning(f"Error closing driver: {e}")
            finally:
                self._driver = None
                self.enabled = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Cleanup on garbage collection."""
        self.close()


# Global connection manager instance
_connection_manager: Optional[Neo4jConnectionManager] = None


def get_neo4j_connection_manager() -> Neo4jConnectionManager:
    """Get or create Neo4j connection manager instance"""
    # Always create new instance to pick up latest env vars
    # TODO: Implement proper singleton with env var refresh
    return Neo4jConnectionManager()