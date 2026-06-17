import json
import time
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple
from sqlalchemy.orm import Session
import google.generativeai as genai

from backend.config import Config
from backend.database.models import QueryLog
from backend.services.vector_store import VectorStoreManager
from backend.utils.logger import get_logger

logger = get_logger("rag_pipeline")

class RAGPipeline:
    """
    Orchestrates the entire Retrieval Augmented Generation (RAG) flow:
    query verification, vector retrieval, out-of-context check,
    LLM invocation, performance tracking, and database logging.
    """

    OUT_OF_CONTEXT_ANSWER = "The answer is not present in the AWS Customer Agreement document."

    @classmethod
    def ask(cls, query: str, db: Session) -> Dict[str, Any]:
        """
        Executes the full RAG process for the given user query.
        """
        start_time = time.time()
        
        # 1. Query Validation
        if not query or not query.strip():
            logger.warning("Received an empty query.")
            return {
                "query": "",
                "answer": "Query cannot be empty.",
                "sources": [],
                "latency_ms": 0.0,
                "answer_found": False
            }

        query = query.strip()
        logger.info(f"Processing query: '{query}'")

        # 2. Check FAISS Vector Store
        vector_store = VectorStoreManager.get_vector_store()
        if vector_store is None:
            logger.error("Vector database is not initialized.")
            raise ValueError("Vector store is missing. Please ingest the PDF document first.")

        # 3. Retrieve chunks (Top K = 5)
        k = Config.TOP_K
        retrieved_docs_with_scores = VectorStoreManager.search(query, k=k)
        retrieved_count = len(retrieved_docs_with_scores)

        # 4. Out-of-Context Detection
        # Check if the highest similarity score meets the threshold
        top_score = retrieved_docs_with_scores[0][1] if retrieved_docs_with_scores else 0.0
        threshold = Config.SIMILARITY_THRESHOLD
        
        logger.info(f"Top retrieval similarity score: {top_score:.4f} (Threshold: {threshold:.4f})")
        
        # Prepare response placeholders
        answer = ""
        answer_found = True
        sources = []

        # Populate source citations
        for doc, score in retrieved_docs_with_scores:
            sources.append({
                "page_num": int(doc.metadata.get("page_num", 0)),
                "section": doc.metadata.get("section", "Unknown"),
                "text": doc.page_content,
                "raw_text": doc.metadata.get("raw_content", doc.page_content),
                "score": round(score, 4)
            })

        if top_score < threshold:
            logger.info("Top score is below similarity threshold. Tagging query as OUT OF CONTEXT.")
            answer = cls.OUT_OF_CONTEXT_ANSWER
            answer_found = False
        else:
            # 5. Build prompt and invoke Gemini 2.5 Flash
            logger.info("Top score meets threshold. Calling Gemini 2.5 Flash...")
            
            # Format retrieved chunks for the prompt context
            context_blocks = []
            for doc, score in retrieved_docs_with_scores:
                context_blocks.append(doc.page_content)
            
            context_str = "\n\n---\n\n".join(context_blocks)
            
            prompt_template = (
                "You are an expert legal document assistant.\n"
                "Answer ONLY from the provided context about the AWS Customer Agreement.\n"
                "If the answer is not available in the context, say exactly:\n"
                "\"The answer is not present in the AWS Customer Agreement document.\"\n"
                "Never hallucinate. Always rely on the provided text.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            )

            # Check for API key configuration
            api_key = Config.GEMINI_API_KEY
            if not api_key or api_key == "your_gemini_api_key_here":
                logger.error("Gemini API key is not configured in .env")
                raise ValueError("Gemini API key is not set. Please set the GEMINI_API_KEY environment variable.")

            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(Config.GEMINI_MODEL_NAME)
                
                # Call LLM
                response = model.generate_content(prompt_template)
                answer = response.text.strip()
                
                # Catch case where LLM claims answer is not found
                if cls.OUT_OF_CONTEXT_ANSWER.lower() in answer.lower():
                    answer_found = False
                    
            except Exception as e:
                logger.error(f"Error calling Gemini API: {e}")
                raise RuntimeError(f"Error during LLM text generation: {e}") from e

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000.0
        latency_ms = round(latency_ms, 2)
        logger.info(f"RAG query finished. Latency: {latency_ms} ms. Answer found: {answer_found}")

        # 6. Database Logging
        try:
            # Serialize sources to store in SQLite
            serialized_sources = json.dumps(sources)
            
            log_entry = QueryLog(
                query=query,
                answer=answer,
                source_chunks=serialized_sources,
                latency_ms=latency_ms,
                retrieved_chunks_count=retrieved_count,
                answer_found=answer_found,
                timestamp=datetime.utcnow()
            )
            
            db.add(log_entry)
            db.commit()
            logger.info("Query successfully logged in SQLite database.")
        except Exception as e:
            logger.error(f"Failed to log query into database: {e}")
            db.rollback()
            # Do not raise error to block user response if database log fails, but log it

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "latency_ms": latency_ms,
            "answer_found": answer_found
        }
