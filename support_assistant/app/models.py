"""
models.py
=========
Pydantic schemas for request and response models.
"""

from typing import List
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., description="The user question to the Zepto support assistant", min_length=1)


class AskResponse(BaseModel):
    answer: str = Field(..., description="The generated response message")
    sources: List[str] = Field(default_factory=list, description="List of source document/chunk IDs used for grounding")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
