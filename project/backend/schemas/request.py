from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """
    Request payload schema for asking a question to the RAG system.
    """
    query: str = Field(..., description="The question related to the AWS Customer Agreement", min_length=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are the AWS Contracting Parties?"
            }
        }
    }
