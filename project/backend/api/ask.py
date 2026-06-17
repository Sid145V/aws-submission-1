from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.schemas.request import QueryRequest
from backend.schemas.response import QueryResponse
from backend.services.rag_pipeline import RAGPipeline
from backend.utils.logger import get_logger

logger = get_logger("api_ask")
router = APIRouter()

@router.post("/ask", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def ask_question(
    request: QueryRequest, 
    db: Session = Depends(get_db)
) -> QueryResponse:
    """
    Question-Answering endpoint. Takes a user question, searches the vector store, 
    evaluates contextual relevance, queries the Gemini model, and returns 
    the result alongside page and section references. Logs execution metrics.
    """
    query_text = request.query.strip()
    if not query_text:
        logger.warning("Query text is empty or whitespace-only.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty."
        )

    try:
        # Run the RAG pipeline
        result = RAGPipeline.ask(query_text, db)
        
        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result["sources"],
            latency_ms=result["latency_ms"]
        )
        
    except ValueError as ve:
        # Vector store or Gemini key missing/uninitialized
        logger.error(f"Value error in RAG pipeline: {ve}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(ve)
        )
    except RuntimeError as re:
        # Gemini API call failure
        logger.error(f"Runtime error in LLM generation: {re}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(re)
        )
    except Exception as e:
        logger.error(f"Unexpected error in Q&A endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
