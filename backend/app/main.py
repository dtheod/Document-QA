from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import logging
import mlflow

from app.core.config import settings
from app.api.routers import chat, documents, evaluation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow experiment that holds the traces of real chat requests (eval runs have their own)
TRACE_EXPERIMENT = "document-qa-app"


def enable_tracing():
    """Trace every workflow run to MLflow, if a tracking server is configured and up."""
    uri = settings.mlflow_tracking_uri
    if not uri:
        return
    try:
        # Fail fast: MLflow's own client retries an unreachable server for minutes
        if uri.startswith("http"):
            httpx.get(f"{uri}/health", timeout=3).raise_for_status()
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(TRACE_EXPERIMENT)
        mlflow.langchain.autolog()
        logger.info(f"Tracing to MLflow at {uri}, experiment '{TRACE_EXPERIMENT}'")
    except Exception as e:
        logger.warning(f"MLflow tracing is off: {uri} is not reachable ({e})")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    enable_tracing()
    yield


app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,  # the API uses no cookies or auth headers
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, tags=["chat"])
app.include_router(documents.router, tags=["documents"])
app.include_router(evaluation.router, tags=["evaluation"])


@app.get("/")
async def root():
    return {"message": "Document QA API"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
