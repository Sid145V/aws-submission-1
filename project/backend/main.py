from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from backend.config import Config
from backend.database.db import engine, Base
from backend.database.models import QueryLog  # Import models to ensure registration
from backend.api import ingest, ask, analytics
from backend.services.vector_store import VectorStoreManager
from backend.utils.logger import get_logger

logger = get_logger("main")

# Initialize FastAPI application
app = FastAPI(
    title="AWS Customer Agreement RAG Assistant",
    description="A production-ready Retrieval Augmented Generation Q&A system for AWS Customer Agreement.",
    version="1.0.0"
)

# Configure CORS middleware
# Allow localhost origins for local frontend connections (Streamlit usually runs on 8501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific origins in strict production environments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(ask.router, tags=["RAG QA"])
app.include_router(analytics.router, tags=["Analytics"])

@app.on_event("startup")
def startup_event():
    """
    Application startup handler. Creates tables and attempts to load the vector store index.
    """
    logger.info("Starting up AWS Customer Agreement RAG Assistant backend...")
    
    # 1. Create SQL database tables automatically
    logger.info("Initializing database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified/created successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize database tables: {e}")
        raise e

    # 2. Attempt to pre-load FAISS vector store
    try:
        db = VectorStoreManager.get_vector_store()
        if db is not None:
            logger.info("Pre-loaded FAISS vector store on startup.")
        else:
            logger.warning("FAISS vector store is not initialized. Please call POST /ingest to process the PDF.")
    except Exception as e:
        logger.error(f"Error loading FAISS vector store on startup: {e}")

@app.get("/")
def read_root():
    """
    Heartbeat endpoint to verify server status.
    """
    vector_store_ready = VectorStoreManager.get_vector_store() is not None
    api_key_configured = bool(Config.GEMINI_API_KEY and Config.GEMINI_API_KEY != "your_gemini_api_key_here")
    
    return {
        "status": "online",
        "service": "AWS Customer Agreement RAG Assistant API",
        "vector_store_initialized": vector_store_ready,
        "gemini_api_configured": api_key_configured,
        "config": {
            "embedding_model": Config.EMBEDDING_MODEL_NAME,
            "llm_model": Config.GEMINI_MODEL_NAME,
            "similarity_threshold": Config.SIMILARITY_THRESHOLD,
            "top_k": Config.TOP_K
        }
    }

if __name__ == "__main__":
    # Start uvicorn server directly if run as a script
    uvicorn.run(
        "backend.main:app",
        host=Config.BACKEND_HOST,
        port=Config.BACKEND_PORT,
        reload=True
    )
