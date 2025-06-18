import logging
from typing import Optional, Dict, Any
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from pinecone import ServerlessSpec, PodSpec
from src.config.settings import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_ENVIRONMENT,
)

logger = logging.getLogger(__name__)

def init_pinecone():
    """Initialize and return a Pinecone client."""
    return PineconeClient(api_key=PINECONE_API_KEY)

def delete_existing_index(pc, index_name: str):
    """Delete an existing Pinecone index if it exists."""
    if index_name in pc.list_indexes().names():
        logger.info(f"Deleting existing index: {index_name}")
        pc.delete_index(index_name)
        return True
    return False

def create_new_index(pc, index_name: str, dimension: int, environment: str):
    """Create a new Pinecone index with the specified parameters."""
    logger.info(f"Creating new index '{index_name}' with dimension {dimension}")
    
    try:
        # For serverless indexes (recommended for most use cases)
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=environment.lower().replace('gcp', 'us-east1')  # Handle different cloud regions
            )
        )
    except Exception as e:
        logger.error(f"Error creating serverless index: {e}")
        logger.info("Falling back to pod-based index...")
        # Fallback to pod-based index if serverless fails
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine"
        )

def get_or_create_index(
    embeddings, 
    index_name: str = PINECONE_INDEX_NAME, 
    environment: str = PINECONE_ENVIRONMENT,
    recreate: bool = True
):
    """
    Get or create a Pinecone index with the correct dimensions.
    
    Args:
        embeddings: The embedding model to use
        index_name: Name of the index
        environment: Cloud environment/region
        recreate: If True, will delete and recreate the index if dimensions don't match
    """
    pc = init_pinecone()
    
    # Get the embedding dimension
    embedding_dimension = len(embeddings.embed_query("test"))
    logger.info(f"Using embedding dimension: {embedding_dimension}")
    
    # Check if index exists
    if index_name in pc.list_indexes().names():
        index = pc.Index(index_name)
        index_stats = index.describe_index_stats()
        
        # Get the dimension from the index stats
        index_dimension = index_stats.dimension
        logger.info(f"Existing index dimension: {index_dimension}")
        
        # If dimensions don't match and recreate is True, delete and recreate
        if index_dimension != embedding_dimension:
            if recreate:
                logger.warning(f"Dimension mismatch ({index_dimension} vs {embedding_dimension}). Recreating index...")
                delete_existing_index(pc, index_name)
                create_new_index(pc, index_name, embedding_dimension, environment)
            else:
                raise ValueError(
                    f"Dimension mismatch: Index has dimension {index_dimension} "
                    f"but model requires {embedding_dimension}. "
                    "Set recreate=True to automatically recreate the index."
                )
    else:
        # Create new index if it doesn't exist
        create_new_index(pc, index_name, embedding_dimension, environment)
    
    return pc.Index(index_name)


if __name__ == "__main__":
    # This allows  to run this script directly for testing
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings()
    vectorstore = get_or_create_index(embeddings)
    print(f"Vectorstore initialized with index: {PINECONE_INDEX_NAME}")
