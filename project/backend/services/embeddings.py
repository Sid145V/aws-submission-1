import google.generativeai as genai
from langchain_core.embeddings import Embeddings
from backend.config import Config
from backend.utils.logger import get_logger
from typing import List

logger = get_logger("embeddings")


class GoogleEmbeddings(Embeddings):
    """
    Lightweight embedding wrapper using Google's free Generative AI embedding API.
    Replaces heavy sentence-transformers model to stay within free tier RAM limits.
    """

    def __init__(self):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        logger.info("Google Generative AI Embeddings initialized.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return result["embedding"]


class EmbeddingModelWrapper:
    _instance = None

    @classmethod
    def get_embeddings_model(cls) -> GoogleEmbeddings:
        if cls._instance is None:
            logger.info("Loading Google Embedding model...")
            cls._instance = GoogleEmbeddings()
            logger.info("Google Embedding model ready.")
        return cls._instance
