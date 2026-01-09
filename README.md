
## Overview

This project provides a robust Document Chatbot utilizing a **FastAPI** backend and a **React (Vite)** frontend. It leverages **LangGraph** for orchestrated agent workflows and **RAGAS** for automated evaluation of retrieval and generation quality.

### Key Features

- 📤 **Multi-Document Upload**: Upload multiple PDF documents (e.g., Apple, Tesla, Uber financial reports) for indexing.
- 💬 **Intelligent Chat**: Ask complex questions and receive context-aware answers citing specific documents.
- 🕵️ **Agentic Workflow**: Uses a graph of agents (Classifier, Retriever, Summarizer, Judge) to handle different query types.
- ⚖️ **Automated Evaluation**: Integrated **RAGAS** suite to audit response quality (faithfulness, relevancy, correctness) against ground truth.
- 📝 **Judge Feedback**: Real-time feedback from a "Judge" agent on the quality of the generated answer.

---

## Architecture

The system uses a sophisticated agent workflow:

![Agent Workflow](backend/data/outputs/workflow_graph.png)

1.  **Router**: Classifies query (General vs. RAG vs. Summarization).
2.  **Retriever**: Fetches relevant chunks from ChromaDB.
3.  **Generator**: Synthesizes answer using context.
4.  **Judge**: Verifies the answer against the retrieved context before returning to the user.

---

## Project Structure

```
Document-QA/
├── backend/                    # FastAPI Backend
│   ├── app/                    # Main application package
│   │   ├── api/                # API layer
│   │   │   └── routers/        # API route handlers
│   │   │       ├── chat.py     # Chat endpoints
│   │   │       ├── documents.py # Document upload
│   │   │       └── evaluation.py # RAGAS evaluation
│   │   ├── core/               # Core configuration
│   │   │   └── config.py       # Environment variables
│   │   ├── models/             # Data models
│   │   │   └── schemas.py      # Pydantic schemas
│   │   ├── services/           # Business logic
│   │   │   ├── workflow/       # LangGraph agents
│   │   │   │   ├── agents.py   # Agent implementations
│   │   │   │   ├── graph.py    # Workflow graph
│   │   │   │   └── chat.py     # Chat service
│   │   │   ├── rag/            # RAG functionality
│   │   │   │   ├── indexing.py # Document indexing
│   │   │   │   └── documents.py # Document service
│   │   │   └── evaluation/     # Evaluation
│   │   │       └── ragas.py    # RAGAS metrics
│   │   ├── main.py             # FastAPI app entry
│   │   └── utils.py            # Shared utilities
│   ├── data/                   # Data storage
│   │   ├── chroma_db/          # Vector database
│   │   ├── doc_store/          # Document store
│   │   └── outputs/            # Generated files
│   ├── prompts/                # YAML prompt templates
│   │   ├── rag_core.yml        # Core RAG prompts
│   │   └── rag_eval.yml        # Evaluation prompts
│   ├── tests/                  # Test files
│   └── pyproject.toml          # Python dependencies
├── frontend/                   # React + Vite Frontend
│   ├── components/             # React components
│   ├── services/               # API client
│   ├── App.tsx                 # Main app component
│   └── package.json            # Node dependencies
├── docker-compose.yml          # Docker orchestration
└── .env                        # Environment variables
```

---

## Prerequisites

Ensure the following are installed:

- **Docker** & **Docker Compose**
- **Python 3.11+** (for local dev)
- **Node.js 18+** (for local dev)

---

## Setup and Run

### 🐳 Using Docker (Recommended)

1.  **Clone the repository**
    ```bash
    git clone <repo-url>
    cd document-qa
    ```

2.  **Environment Setup**
    - Create a `.env` file in the root directory.
    - Add your API keys:
      ```env
      OPENAI_API_KEY=your_key_here
      OPENAI_MODEL_NAME=gpt-4o
      EMBEDDING_MODEL=text-embedding-3-small
      HF_TOKEN=your_huggingface_token_if_needed
      ```

3.  **Run the Application**
    ```bash
    docker-compose up --build
    ```

4.  **Access the App**
    - **Frontend**: [http://localhost:5173](http://localhost:5173)
    - **Backend API**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 💻 Local Development

**Backend**:
```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

**Frontend**:
```bash
cd frontend
npm install
npm run dev
```

---

## Evaluation Suite

The project includes a built-in evaluation tab to run **RAGAS** metrics.

1.  Navigate to the **Evaluations** tab in the UI.
2.  Click **"Start Document Audit"**.
3.  View detailed metrics:
    - **Faithfulness**: Is the answer derived from context?
    - **Answer Correctness**: Does it match ground truth?
    - **Context Recall**: Did we retrieve all necessary info?

---

## Demo

<video src="backend/outputs/screen_docuAIapp.mov" controls="controls" style="max-width: 100%;">
  Your browser does not support the video tag.
</video>

---
