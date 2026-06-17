from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime
from typing import Dict, Any, List
from backend.database.models import QueryLog
from backend.utils.logger import get_logger

logger = get_logger("analytics_service")

class AnalyticsService:
    """
    Service to fetch and compute database analytics for the QueryLog model.
    """

    @classmethod
    def get_analytics(cls, db: Session) -> Dict[str, Any]:
        """
        Calculates all required analytics from the query_logs table.
        """
        logger.info("Computing RAG analytics...")
        
        # 1. Total Questions
        total_questions = db.query(QueryLog).count()
        if total_questions == 0:
            return {
                "total_questions": 0,
                "avg_response_time_ms": 0.0,
                "unanswered_questions_count": 0,
                "success_rate": 0.0,
                "daily_usage": [],
                "top_queries": [],
                "retrieved_chunks_stats": {
                    "avg_chunks_retrieved": 0.0
                }
            }

        # 2. Average Response Time
        avg_latency = db.query(func.avg(QueryLog.latency_ms)).scalar() or 0.0

        # 3. Questions With No Answers
        no_answers_count = db.query(QueryLog).filter(QueryLog.answer_found == False).count()

        # 4. Success Rate
        success_rate = ((total_questions - no_answers_count) / total_questions) * 100.0

        # 5. Daily Usage
        # SQLite: strftime('%Y-%m-%d', timestamp) or func.date(timestamp)
        # Using func.date(QueryLog.timestamp) is highly portable for SQLite
        daily_usage_query = (
            db.query(
                func.date(QueryLog.timestamp).label("date"),
                func.count(QueryLog.id).label("count")
            )
            .group_by("date")
            .order_by("date")
            .all()
        )
        daily_usage = [{"date": r.date, "count": r.count} for r in daily_usage_query]

        # 6. Top 10 Queries (Most Frequently Asked Questions)
        top_queries_query = (
            db.query(
                QueryLog.query,
                func.count(QueryLog.id).label("frequency"),
                func.avg(QueryLog.latency_ms).label("avg_latency_ms")
            )
            .group_by(QueryLog.query)
            .order_by(desc("frequency"))
            .limit(10)
            .all()
        )
        top_queries = [
            {
                "query": r.query,
                "frequency": r.frequency,
                "avg_latency_ms": round(float(r.avg_latency_ms), 2)
            }
            for r in top_queries_query
        ]

        # 7. Retrieval Statistics
        avg_chunks_retrieved = db.query(func.avg(QueryLog.retrieved_chunks_count)).scalar() or 0.0

        analytics_data = {
            "total_questions": total_questions,
            "avg_response_time_ms": round(float(avg_latency), 2),
            "unanswered_questions_count": no_answers_count,
            "success_rate": round(success_rate, 2),
            "daily_usage": daily_usage,
            "top_queries": top_queries,
            "retrieved_chunks_stats": {
                "avg_chunks_retrieved": round(float(avg_chunks_retrieved), 2)
            }
        }
        
        logger.info(f"Analytics computed successfully. Total queries: {total_questions}")
        return analytics_data
