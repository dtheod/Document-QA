from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings
from app.models.schemas import Document
from langchain_classic.storage import LocalFileStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_core.documents import Document as LCDocument
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc import DoclingDocument
from functools import lru_cache
from pathlib import Path
import hashlib
import tiktoken
import uuid

# backend/data/
DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# Chunking strategy used when indexing:
# - "docling": docling's HybridChunker splits along the document's sections and tables; each
#   chunk is embedded together with its section headings and returned as-is. Tables are written
#   as "row, column = value" triplets
# - "docling_markdown": the same, with tables written as Markdown tables (one header row per table)
# - "recursive": fixed-size parent and child chunks of the PyPDF page text; children are
#   embedded and searched, their parents are returned
# recursive had the best retrieval on the eval set (notebooks/retrieval_check.ipynb, MLflow
# experiment "document-qa-retrieval"); the docling chunkers are kept for comparison
CHUNKER = "recursive"

# Configuration for the docling chunker
DOCLING_MAX_TOKENS = 512
DOCLING_TOKEN_ENCODING = "cl100k_base"  # tokenizer of OpenAI's text-embedding-3 models
DOCLING_TABLE_FORMATS = {"docling": "triplets", "docling_markdown": "markdown"}

# Configuration for Text Splitters
PARENT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=200, add_start_index=True
)
CHILD_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=50, add_start_index=True
)


def chunking_params(chunker: str = CHUNKER) -> dict:
    """Settings of a chunking strategy, for experiment tracking."""
    if chunker in DOCLING_TABLE_FORMATS:
        return {
            "chunker": chunker,
            "max_tokens": DOCLING_MAX_TOKENS,
            "token_encoding": DOCLING_TOKEN_ENCODING,
            "table_format": DOCLING_TABLE_FORMATS[chunker],
        }
    if chunker == "recursive":
        return {
            "chunker": chunker,
            "parent_chunk_size": PARENT_SPLITTER._chunk_size,
            "parent_chunk_overlap": PARENT_SPLITTER._chunk_overlap,
            "child_chunk_size": CHILD_SPLITTER._chunk_size,
            "child_chunk_overlap": CHILD_SPLITTER._chunk_overlap,
        }
    return {"chunker": chunker}


def build_retriever(data_dir: Path):
    """Retriever over the vector store and parent doc store persisted in data_dir."""
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )

    # Initialize vectorstore and persistent docstore
    vector_store = Chroma(
        embedding_function=embeddings,
        collection_name="companies_financials_pdfs",
        persist_directory=str(data_dir / "chroma_db"),
    )

    # Filesystem storage for parent documents
    fs = LocalFileStore(str(data_dir / "doc_store"))

    # EncoderBackedStore handles (Document -> bytes) serialization via pickle
    store = create_kv_docstore(fs)

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=CHILD_SPLITTER,
        parent_splitter=PARENT_SPLITTER,
    )
    return retriever, vector_store


@lru_cache(maxsize=1)
def get_retriever():
    """The app's retriever, persisted in backend/data/."""
    return build_retriever(DATA_DIR)


@lru_cache(maxsize=1)
def _pdf_converter():
    # docling's converter needs torch, so it is an optional install
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as e:
        raise RuntimeError(
            "The docling chunkers need docling: run `uv sync --group experiments` in backend/"
        ) from e

    # The PDFs carry a text layer, so OCR would only slow conversion down
    options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def convert_pdf(path: str) -> DoclingDocument:
    return _pdf_converter().convert(path).document


class MarkdownTablesProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc, table_serializer=MarkdownTableSerializer()
        )


def docling_chunks(
    dl_doc: DoclingDocument, table_format: str = "triplets"
) -> list[LCDocument]:
    """Section-aware chunks, each prefixed with its section headings."""
    chunker = HybridChunker(
        tokenizer=OpenAITokenizer(
            tokenizer=tiktoken.get_encoding(DOCLING_TOKEN_ENCODING),
            max_tokens=DOCLING_MAX_TOKENS,
        ),
        **(
            {"serializer_provider": MarkdownTablesProvider()}
            if table_format == "markdown"
            else {}
        ),
    )
    chunks = []
    for chunk in chunker.chunk(dl_doc):
        text = chunker.contextualize(chunk)
        if not text.strip():
            continue
        metadata = {"section": " > ".join(chunk.meta.headings or [])}
        pages = [prov.page_no for item in chunk.meta.doc_items for prov in item.prov]
        if pages:
            # Docling pages are 1-based; store 0-based like PyPDF so citations work for both
            metadata["page"] = min(pages) - 1
        chunks.append(LCDocument(page_content=text, metadata=metadata))
    return chunks


def index_documents(document: Document, chunker: str = CHUNKER, index=None):
    """Index a PDF. `index` is a (retriever, vector_store) pair; defaults to the app's."""
    retriever, vector_store = index or get_retriever()

    # Use document name for metadata if available, else derive it
    company_name = "general"
    if document.document_name:
        # simple heuristic, or trust the input
        company_name = document.document_name.lower().replace(".pdf", "")

    if chunker == "recursive":
        loader = PyPDFLoader(file_path=document.document_path, extraction_mode="plain")

        documents = loader.load()

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
                "chunker": chunker,
            }
            doc.metadata.update(doc_metadata)
            metadata_documents.append(doc)
    elif chunker in DOCLING_TABLE_FORMATS:
        chunks = docling_chunks(
            convert_pdf(document.document_path), DOCLING_TABLE_FORMATS[chunker]
        )
        for chunk in chunks:
            chunk.metadata.update(
                {
                    "company": company_name,
                    "source": document.document_name,
                    "id": str(uuid.uuid4()),
                    "chunker": chunker,
                }
            )
    else:
        raise ValueError(f"Unknown chunker: {chunker}")

    # Re-upload replaces the previous copy: drop its child chunks and parent docs
    existing = vector_store._collection.get(
        where={"source": document.document_name}, include=["metadatas"]
    )
    if existing["ids"]:
        parent_ids = {m["doc_id"] for m in existing["metadatas"] if m.get("doc_id")}
        vector_store.delete(ids=existing["ids"])
        retriever.docstore.mdelete(list(parent_ids))

    if chunker == "recursive":
        retriever.add_documents(metadata_documents)
    else:
        # Docling chunks are already sized for embedding, so each chunk is both the
        # searched child and the returned parent
        retriever.vectorstore.add_documents(
            [
                LCDocument(
                    page_content=chunk.page_content,
                    metadata={**chunk.metadata, retriever.id_key: chunk.metadata["id"]},
                )
                for chunk in chunks
            ]
        )
        retriever.docstore.mset([(chunk.metadata["id"], chunk) for chunk in chunks])

    return None
