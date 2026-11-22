import yaml
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import os


def load_prompt_from_yaml(yml_type: str, prompt_id: str) -> ChatPromptTemplate:
    """Load a prompt definition from a YAML file."""
    # Navigate from app/utils.py to prompts/
    prompt_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "prompts", yml_type
    )
    with open(prompt_path, "r") as f:
        prompts = yaml.safe_load(f)

    for prompt_def in prompts:
        if prompt_def["id"] == prompt_id:
            return ChatPromptTemplate.from_template(prompt_def["template"])

    raise ValueError(f"Prompt with id {prompt_id} not found in {prompt_path}")


def get_company_docs(vectorstore: Chroma, company: str) -> list[Document]:
    # Use Chroma's underlying collection to fetch by metadata
    raw = vectorstore._collection.get(
        where={"company": company},
        include=["documents", "metadatas"],
    )

    docs: list[Document] = []
    for text, meta in zip(raw["documents"], raw["metadatas"]):
        docs.append(Document(page_content=text, metadata=meta or {}))

    return docs


def get_indexed_companies(vectorstore: Chroma) -> list[str]:
    """Distinct company tags of everything in the vector store."""
    raw = vectorstore._collection.get(include=["metadatas"])
    return sorted({m["company"] for m in raw["metadatas"] if m and m.get("company")})


def page_number(metadata: dict) -> int | None:
    """1-based page number (PyPDF records a 0-based `page`)."""
    page = metadata.get("page")
    return page + 1 if isinstance(page, int) else None


def citation_label(metadata: dict) -> str:
    source = metadata.get("source", "unknown")
    page = page_number(metadata)
    return f"[{source}, p. {page}]" if page else f"[{source}]"
