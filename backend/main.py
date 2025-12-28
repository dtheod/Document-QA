from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import logging
from backend.schemas import Document
from backend.indexing import index_documents
from backend.workflow import process_workflow_query

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.getcwd(), "temp_docs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []

class EvaluationRequest(BaseModel):
    context: str = "" # Optional, might rely on indexed docs

@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    try:
        uploaded_files = []
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"File saved to {file_path}")

            # Indexing
            doc = Document(
                document_path=file_path,
                document_name=file.filename
            )
            index_documents(doc)
            logger.info(f"Indexing complete for {file.filename}")
            
            uploaded_files.append({
                "filename": file.filename,
                "status": "indexed"
            })

        # Generate Summary (Reusing workflow or specific agent?)
        # For now, we return a generic message as summarizing multiple docs requires a different approach
        summary = f"Successfully indexed {len(uploaded_files)} documents."

        return {
            "message": "Files uploaded and indexed successfully",
            "summary": summary,
            "files": uploaded_files
        }

    except Exception as e:
        logger.error(f"Error uploading files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Convert history to format expected by workflow if needed
        # Assuming frontend sends [{role: 'user', content: '...'}, ...] matches
        
        result = process_workflow_query(request.query, request.history)
        answer = result.get("response", "No answer generated.")
        
        return {
            "answer": answer,
            "judge_result": result.get("judge_result"),
            "judge_feedback": result.get("judge_feedback")
        }
    except Exception as e:
        logger.error(f"Error in chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

from backend.evaluation import run_ragas_evaluation

@app.post("/evaluate")
async def evaluate_endpoint(request: EvaluationRequest):
    try:
        # Trigger the full RAGAS evaluation suite
        # Note: This ignores the 'request.context' or specific document context for now
        # and runs the global verification suite defined in backend/evaluation.py
        
        results = await run_ragas_evaluation()
        return results

    except Exception as e:
         logger.error(f"Error in evaluation: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
