import os
from typing import List, Tuple, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from backend.config import Config
from backend.services.embeddings import EmbeddingModelWrapper
from backend.utils.logger import get_logger

logger = get_logger("vector_store")

class VectorStoreManager:
    """
    Service to manage FAISS vector store creation, persistence, loading,
    and similarity search operations.
    """
    
    _db_instance: Optional[FAISS] = None

    @classmethod
    def get_vector_store(cls) -> Optional[FAISS]:
        """
        Retrieves the active FAISS instance. Returns None if not loaded/created.
        """
        if cls._db_instance is None:
            # Try to load index from disk if it exists
            index_path = Config.VECTOR_STORE_DIR / "faiss_index.faiss"
            if os.path.exists(index_path):
                logger.info(f"Loading FAISS index from: {index_path}")
                try:
                    embeddings = EmbeddingModelWrapper.get_embeddings_model()
                    cls._db_instance = FAISS.load_local(
                        folder_path=str(Config.VECTOR_STORE_DIR),
                        index_name="faiss_index",
                        embeddings=embeddings,
                        allow_dangerous_deserialization=True  # Required for loading local FAISS pickle files
                    )
                    logger.info("FAISS index loaded successfully from disk.")
                except Exception as e:
                    logger.error(f"Error loading FAISS index: {e}")
                    cls._db_instance = None
            else:
                logger.info("No existing FAISS index found on disk.")
        return cls._db_instance

    @classmethod
    def create_and_save_index(cls, chunks: List[Document]) -> FAISS:
        """
        Creates a new FAISS vector store from document chunks and persists it to disk.
        """
        logger.info(f"Creating FAISS index with {len(chunks)} chunks...")
        embeddings = EmbeddingModelWrapper.get_embeddings_model()
        
        try:
            # Build index
            db = FAISS.from_documents(chunks, embeddings)
            
            # Save index to disk
            db.save_local(
                folder_path=str(Config.VECTOR_STORE_DIR),
                index_name="faiss_index"
            )
            
            # Update active instance
            cls._db_instance = db
            logger.info(f"Successfully saved FAISS index to {Config.VECTOR_STORE_DIR}")
            return db
        except Exception as e:
            logger.error(f"Failed to create and save FAISS index: {e}")
            raise e

    @classmethod
    def search(cls, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Performs a similarity search in the FAISS index.
        Converts default FAISS squared L2 distance to Cosine Similarity score.
        Returns a list of tuples: (Document, cosine_similarity_score).
        """
        db = cls.get_vector_store()
        if db is None:
            logger.error("Attempted to search but FAISS index is not initialized.")
            raise ValueError("Vector store is not initialized. Please run ingestion first.")

        # FAISS similarity_search_with_score returns L2 distance squared
        results = db.similarity_search_with_score(query, k=k)
        
        processed_results = []
        for doc, score in results:
            # Convert L2 distance squared to Cosine Similarity
            # Cosine_Sim = 1 - (L2_dist_sq / 2)
            cosine_similarity = 1.0 - (float(score) / 2.0)
            
            # Cap values between 0.0 and 1.0
            cosine_similarity = max(0.0, min(1.0, cosine_similarity))
            
            processed_results.append((doc, cosine_similarity))
            
        logger.info(f"Search query: '{query}' yielded {len(processed_results)} results. Top score: {processed_results[0][1] if processed_results else 0.0:.4f}")
        return processed_results
