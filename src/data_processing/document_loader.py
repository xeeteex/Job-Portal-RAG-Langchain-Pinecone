# @ IMPORTING NECESSARY LIBRARIES
from langchain_community.document_loaders import DirectoryLoader
from src.config.settings import DATA_DIR
import logging

# @ SET LOGGING LEVEL
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents(file_pattern="**/*.pdf"):
    """
    Load documents as single chunks (no splitting).
    Each file will be a single document chunk.
    """
    try:
        logger.info(f"Loading documents from {DATA_DIR}")
        loader = DirectoryLoader(DATA_DIR, glob=file_pattern)
        documents = loader.load()
        
        # Add metadata to identify document type
        for doc in documents:
            doc.metadata["is_whole_document"] = True
            
        logger.info(f"Loaded {len(documents)} documents as single chunks")
        return documents
        
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise

# Alias for backward compatibility
load_cv_documents = load_documents

if __name__ == "__main__":
    # Test loading documents as single chunks
    print("Loading documents as single chunks:")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    if docs:
        print(f"First document content length: {len(docs[0].page_content)} characters")
