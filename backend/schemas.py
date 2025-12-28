from pydantic import BaseModel, Field
from typing import Optional

class Document(BaseModel):
    document_path: str = Field(..., description="Path to the document file")
    document_name: str = Field(..., description="Name of the document")
    document_id: Optional[str] = Field(None, description="Unique identifier for the document")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
