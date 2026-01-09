from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import logging
from app.models.schemas import Document
from app.services.rag.indexing import index_documents

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = os.path.join(os.getcwd(), "temp_docs")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    try:
        uploaded_files = []
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            logger.info(f"File saved to {file_path}")

            # Indexing
            doc = Document(document_path=file_path, document_name=file.filename)
            index_documents(doc)
            logger.info(f"Indexing complete for {file.filename}")

            uploaded_files.append({"filename": file.filename, "status": "indexed"})

        # Generate Summary (Reusing workflow or specific agent?)
        # For now, we return a generic message as summarizing multiple docs requires a different approach
        summary = f"Successfully indexed {len(uploaded_files)} documents."

        return {
            "message": "Files uploaded and indexed successfully",
            "summary": summary,
            "files": uploaded_files,
        }

    except Exception as e:
        logger.error(f"Error uploading files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
