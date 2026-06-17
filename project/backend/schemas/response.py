from pydantic import BaseModel, Field
from typing import List, Dict, Any

class SourceChunk(BaseModel):
    """
    Schema representing a cited source text chunk from the PDF.
    """
    page_num: int = Field(..., description="The page number where the chunk resides")
    section: str = Field(..., description="The section name of the document")
    text: str = Field(..., description="The text content of the chunk with section context prepended")
    raw_text: str = Field(..., description="The raw page text snippet of this chunk")
    score: float = Field(..., description="Cosine similarity score for the match")

class QueryResponse(BaseModel):
    """
    Response schema returned by the /ask endpoint.
    """
    query: str = Field(..., description="The original search query")
    answer: str = Field(..., description="The generated text answer from Gemini or the fallback out-of-context response")
    sources: List[SourceChunk] = Field(..., description="List of document text segments cited to answer the query")
    latency_ms: float = Field(..., description="Total time taken to retrieve and generate the response in milliseconds")

class DailyUsageItem(BaseModel):
    """
    Schema for daily volume tracking.
    """
    date: str = Field(..., description="The date of query execution (YYYY-MM-DD)")
    count: int = Field(..., description="Number of queries processed on this date")

class TopQueryItem(BaseModel):
    """
    Schema representing frequently asked questions.
    """
    query: str = Field(..., description="The exact question text")
    frequency: int = Field(..., description="Number of times this question was asked")
    avg_latency_ms: float = Field(..., description="Average response latency for this query")

class RetrievalStats(BaseModel):
    """
    Schema for retrieval analytics.
    """
    avg_chunks_retrieved: float = Field(..., description="Average number of chunks retrieved per search")

class AnalyticsResponse(BaseModel):
    """
    Structured analytics payload returned by the /analytics endpoint.
    """
    total_questions: int = Field(..., description="Total number of logged user queries")
    avg_response_time_ms: float = Field(..., description="Average response time across all queries")
    unanswered_questions_count: int = Field(..., description="Count of queries that were determined out-of-context")
    success_rate: float = Field(..., description="Percentage of queries successfully answered")
    daily_usage: List[DailyUsageItem] = Field(..., description="Time-series log of query volume per day")
    top_queries: List[TopQueryItem] = Field(..., description="List of top unique queries by frequency")
    retrieved_chunks_stats: RetrievalStats = Field(..., description="Stats on text chunks retrieval counts")
