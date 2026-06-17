from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from backend.config import Config

import os

DATABASE_URL = Config.DATABASE_URL

# Ensure parent directory for SQLite database file exists
db_dir = os.path.dirname(DATABASE_URL.replace("sqlite:///", ""))
if db_dir and not os.path.exists(db_dir):
    os.makedirs(db_dir, exist_ok=True)

# Create SQLAlchemy engine (check_same_thread is needed for SQLite to run in FastAPI multi-threaded context)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for SQLAlchemy models
Base = declarative_base()

def get_db():
    """
    Dependency helper that yields a database session and closes it on completion.
    Used in FastAPI routes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
