from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.config import Config
from backend.utils.logger import get_logger

logger = get_logger("embeddings")

class EmbeddingModelWrapper:
    """
    Wrapper around the Sentence Transformers embedding model used for generating 
    text embeddings. Implements the LangChain Embeddings interface.
    """
    
    _instance = None

    @classmethod
    def get_embeddings_model(cls) -> HuggingFaceEmbeddings:
        """
        Singleton pattern to load and cache the HuggingFace embedding model.
        """
        if cls._instance is None:
            model_name = Config.EMBEDDING_MODEL_NAME
            logger.info(f"Initializing HuggingFaceEmbeddings with model: {model_name}")
            try:
                # Initialize HuggingFaceEmbeddings using SentenceTransformers under the hood
                cls._instance = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},  # Default to CPU for portability
                    encode_kwargs={'normalize_embeddings': True}  # Use Cosine Similarity / Inner Product
                )
                logger.info("Embedding model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise e
        return cls._instance
