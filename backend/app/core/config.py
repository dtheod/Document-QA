"""App settings, read from environment variables or the repo's .env file."""

from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Document-QA/.env; in Docker the variables come from the container environment instead
ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    openai_api_key: str
    openai_model_name: str = "gpt-6-luna"
    embedding_model: str = "text-embedding-3-small"
    hf_token: str | None = None  # only needed by the injection check
    # MLflow server that receives a trace of every workflow run; tracing is off when unset
    mlflow_tracking_uri: str | None = None
    # Comma-separated list of origins allowed to call the API
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_origins(cls, value):
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


settings = Settings()
