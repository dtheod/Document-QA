from pydantic import BaseModel, Field
from typing import Optional, TypedDict


class Document(BaseModel):
    document_path: str = Field(..., description="Path to the document file")
    document_name: str = Field(..., description="Name of the document")
    document_id: Optional[str] = Field(
        None, description="Unique identifier for the document"
    )
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class SourceReference(BaseModel):
    document_id: str
    document_name: str
    relevance_score: float
    content_excerpt: str


class ChatResponseModel(BaseModel):
    response: str
    sources: list[SourceReference]
    judge_result: bool
    judge_feedback: str


class ClassificationOutput(BaseModel):
    company: str = Field(..., description="The company the user is asking about.")
    query_type: str = Field(
        ..., description="The intent of the user (rag, summarize, unknown)."
    )


class JudgeOutput(BaseModel):
    is_valid: bool = Field(
        ...,
        description="True if the answer is accurate, valid and helpful, False otherwise.",
    )
    feedback: str = Field(
        ..., description="Detailed feedback on why the answer is valid or invalid."
    )


class State(TypedDict):
    query: str
    query_type: str
    query_safety_type: str
    company: str
    retriever_results: str
    response: str
    judge_result: bool
    judge_feedback: str
    chat_history: list[dict]
