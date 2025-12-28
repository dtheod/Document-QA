import asyncio
import logging
from backend.schemas import Document
from backend.indexing import index_documents

log = logging.getLogger(__name__)


async def upload_document(document: Document) -> bool:
    """
    Service to upload a document into the vector store.

    Args:
        document (Document): The document object containing metadata and
            content to be uploaded.

    Returns:
        bool: True if the document was successfully uploaded. False otherwise.
    """
    log.info(
        f"Start uploading the Document id: {document.document_id} | "
        f"name: {document.document_name} | "
        f"path: {document.document_path}"
    )

    try:
        await asyncio.sleep(1)
        # Indexing documents. Keeping it modular for debugging purposes
        _ = index_documents(document)        
        log.info(f"Completed to upload Document {document.document_name} ")
        return True

    except (OSError, ValueError) as e:
        log.error(f"Failed to upload document {document.document_name}: {str(e)}")
        raise
