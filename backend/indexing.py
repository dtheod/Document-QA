
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from backend.schemas import Document
from langchain_classic.storage import LocalFileStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.storage._lc_store import create_kv_docstore
import os
import chromadb
from functools import lru_cache

@lru_cache(maxsize=1)
def get_retriever():
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"),
                                  api_key=os.getenv("OPENAI_API_KEY"))

    # Initialize text splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)

    # Persistence paths
    persist_db = os.path.join(os.path.dirname(__file__), "chroma_db")
    persist_doc_store = os.path.join(os.path.dirname(__file__), "doc_store")

    # Initialize vectorstore and persistent docstore
    vector_store = Chroma(embedding_function=embeddings,
                          collection_name="companies_financials_pdfs",
                          persist_directory=persist_db)
    
    # Filesystem storage for parent documents
    fs = LocalFileStore(persist_doc_store)
    
    # EncoderBackedStore handles (Document -> bytes) serialization via pickle
    store = create_kv_docstore(fs)

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    return retriever, vector_store 

def index_documents(document: Document):
    retriever, vector_store = get_retriever()
    
    loader = PyPDFLoader(file_path=document.document_path,
                         extraction_mode="plain")
    
    documents = loader.load()
    
    # Use document name for metadata if available, else derive it
    company_name = "general"
    if document.document_name:
         # simple heuristic, or trust the input
         company_name = document.document_name.lower().replace(".pdf", "")

    metadata_documents = []
    # Create unique IDs based on file name to avoid collisions
    ids = [f"{company_name}_doc_{i}" for i in range(len(documents))]
    
    for idx, doc in enumerate(documents):
        # Attach metadata
        doc_metadata = {"company": company_name, "source": document.document_name, "id": ids[idx]}
        doc.metadata.update(doc_metadata)
        metadata_documents.append(doc)

    retriever.add_documents(metadata_documents)    

    return None