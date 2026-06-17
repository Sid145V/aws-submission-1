from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.schemas.response import AnalyticsResponse
from backend.services.analytics_service import AnalyticsService
from backend.utils.logger import get_logger

logger = get_logger("api_analytics")
router = APIRouter()

@router.get("/analytics", response_model=AnalyticsResponse, status_code=status.HTTP_200_OK)
async def get_system_analytics(db: Session = Depends(get_db)) -> AnalyticsResponse:
    """
    Analytics endpoint. Calculates and returns system utilization statistics:
    total questions, success rates, unanswered question ratios, latency averages, 
    daily query charts, and top queries.
    """
    try:
        analytics_data = AnalyticsService.get_analytics(db)
        return AnalyticsResponse(**analytics_data)
    except Exception as e:
        logger.error(f"Failed to fetch system analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching analytics: {str(e)}"
        )
