from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from backend.database.db import Base

class QueryLog(Base):
    """
    SQLAlchemy model representing logs of RAG queries, answers, latency,
    and metadata for analysis.
    """
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    query = Column(String, nullable=False, index=True)
    answer = Column(Text, nullable=False)
    source_chunks = Column(Text, nullable=True)  # Stored as a JSON-serialized string
    latency_ms = Column(Float, nullable=False)
    retrieved_chunks_count = Column(Integer, nullable=False)
    answer_found = Column(Boolean, nullable=False, default=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<QueryLog id={self.id} query='{self.query[:20]}...' answer_found={self.answer_found}>"
