from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from app.models.schemas import Document
from langchain_classic.storage import LocalFileStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.storage._lc_store import create_kv_docstore
import os
from functools import lru_cache
import hashlib

# Configuration for Text Splitters
PARENT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=200, add_start_index=True
)
CHILD_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=50, add_start_index=True
)


@lru_cache(maxsize=1)
def get_retriever():
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
    )

    # Persistence paths - navigate from app/services/rag/ to backend/data/
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )
    persist_db = os.path.join(base_dir, "data", "chroma_db")
    persist_doc_store = os.path.join(base_dir, "data", "doc_store")

    # Initialize vectorstore and persistent docstore
    vector_store = Chroma(
        embedding_function=embeddings,
        collection_name="companies_financials_pdfs",
        persist_directory=persist_db,
    )

    # Filesystem storage for parent documents
    fs = LocalFileStore(persist_doc_store)

    # EncoderBackedStore handles (Document -> bytes) serialization via pickle
    store = create_kv_docstore(fs)

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=CHILD_SPLITTER,
        parent_splitter=PARENT_SPLITTER,
    )
    return retriever, vector_store


def index_documents(document: Document):
    retriever, vector_store = get_retriever()

    loader = PyPDFLoader(file_path=document.document_path, extraction_mode="plain")

    documents = loader.load()

    # Use document name for metadata if available, else derive it
    company_name = "general"
    if document.document_name:
        # simple heuristic, or trust the input
        company_name = document.document_name.lower().replace(".pdf", "")

    # Create unique IDs based on content hash to avoid duplicates
    # and collision if file is re-uploaded
    ids = []
    metadata_documents = []

    for doc in documents:
        # Create deterministic ID based on filename and content
        file_name = document.document_name
        content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
        unique_id = f"{file_name}_{content_hash}"

        ids.append(unique_id)

        # Attach metadata
        doc_metadata = {
            "company": company_name,
            "source": document.document_name,
            "id": unique_id,
        }
        doc.metadata.update(doc_metadata)
        metadata_documents.append(doc)

    retriever.add_documents(metadata_documents)

    return None
