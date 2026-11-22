import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# HF_TOKEN is only needed by the injection check, which is disabled for now
HAS_API_KEYS = bool(os.getenv("OPENAI_API_KEY"))

# The app requires an OpenAI key and builds its clients at import time; a placeholder
# lets the offline tests import it. No request is ever sent with it.
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "test-key"


def pytest_collection_modifyitems(config, items):
    if HAS_API_KEYS:
        return
    skip = pytest.mark.skip(reason="needs OPENAI_API_KEY in .env")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
