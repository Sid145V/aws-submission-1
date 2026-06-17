from fastapi import APIRouter, File, UploadFile, HTTPException, status
from typing import Optional, Dict, Any
import os
import shutil

from backend.config import Config
from backend.services.pdf_loader import PDFLoader
from backend.services.chunker import DocumentChunker
from backend.services.vector_store import VectorStoreManager
from backend.utils.logger import get_logger

logger = get_logger("api_ingest")
router = APIRouter()

@router.post("/ingest", status_code=status.HTTP_201_CREATED)
async def ingest_pdf(file: Optional[UploadFile] = File(None)) -> Dict[str, Any]:
    """
    Ingests the AWS Customer Agreement PDF:
    1. Saves the uploaded PDF file (or falls back to the file on disk).
    2. Extracts and cleans text page-by-page.
    3. Groups content by sections to maintain hierarchy.
    4. Chunks the text into segments.
    5. Computes vector embeddings and saves the FAISS index to disk.
    """
    logger.info("Ingest PDF endpoint invoked.")
    target_path = Config.DATA_DIR / "aws_customer_agreement.pdf"
    
    # 1. Save uploaded file if provided
    if file is not None:
        if not file.filename.endswith(".pdf"):
            logger.warning(f"Uploaded file '{file.filename}' is not a PDF.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported."
            )
        
        try:
            logger.info(f"Saving uploaded file '{file.filename}' to {target_path}")
            with open(target_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error(f"Failed to save uploaded PDF file: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save uploaded PDF: {str(e)}"
            )
    else:
        # Fallback to local file
        if not os.path.exists(target_path):
            logger.error(f"No PDF found on disk at {target_path} and no file was uploaded.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No PDF file was uploaded and default 'aws_customer_agreement.pdf' is missing in project data folder."
            )
        logger.info(f"Processing existing local PDF file: {target_path}")

    # 2. Extract, clean, and chunk PDF contents
    try:
        pages = PDFLoader.load_pdf(str(target_path))
        chunks = DocumentChunker.chunk_pages(pages)
        
        if not chunks:
            logger.warning("No text chunks were generated from the PDF.")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to extract any readable content or sections from the PDF."
            )

        # 3. Create FAISS Index and Save to disk
        VectorStoreManager.create_and_save_index(chunks)
        
        logger.info("PDF Ingestion completed successfully.")
        return {
            "status": "success",
            "message": "AWS Customer Agreement PDF ingested and vector index saved successfully.",
            "pages_processed": len(pages),
            "chunks_created": len(chunks),
            "file_name": file.filename if file else "aws_customer_agreement.pdf"
        }
        
    except FileNotFoundError as fnfe:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnfe))
    except Exception as e:
        logger.error(f"Error during PDF ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during ingestion: {str(e)}"
        )
