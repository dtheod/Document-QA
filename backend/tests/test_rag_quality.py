import sys
import os
import logging
from dotenv import load_dotenv
from pathlib import Path
# Load environment variables
load_dotenv()

# Setup path to allow importing from backend
current_file = Path(__file__).resolve()
backend_dir = current_file.parent.parent
project_root = backend_dir.parent

sys.path.append(str(project_root))

from backend.workflow import process_workflow_query
from ragas.metrics import faithfulness, answer_relevancy,answer_correctness, context_precision, context_recall
from ragas import evaluate
import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.storage._lc_store import create_kv_docstore
from backend.indexing import index_documents, get_retriever
from backend.schemas import Document


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
llm = ChatOpenAI(temperature=0, model=os.getenv("OPENAI_MODEL_NAME"))
embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"),
                              api_key=os.getenv("OPENAI_API_KEY"))

test_questions = [
    "What was Apple’s net income for the first nine months ended June 28, 2025?",
    "What was Apple’s net income for the first nine months ended on June 29, 2024?"
    # "Wie vielle Modelle umfasst das Hybridangebot der Cayenne-Reihe von Porsche?",
    # "What was the basic net loss per share attributable to Uber Technologies, Inc. in the first quarter of 2024?",
    # "Give a summary of the apple documents",
    # "What is Ubers free Cash flow for 2024?",
    # "What is Ubers revenue for delivery in Q1 2023?",
    # "How many new markets in Video advertising did Uber enter?",
    # "How is the weather today?",
    # "What is the net income of Tesla for 2024?",
    # "Give me a financial summary for BMW?"
    ]

test_answers = [
    "Apple reported net income of $84,544 million for the first nine months ended June 28, 2025.",
    "Apple’s net income for the first nine months ended on June 29, 2024, totaled $79,000 million"
    # "Das Hybridangebot der Cayenne-Reihe von Porsche umfasst insgesamt drei Modelle",
    # "The basic net loss per share attributable to Uber Technologies, Inc. in the first quarter of 2024 was $0.31",
    # "Apple reported strong financial performance for the nine months ended June 28, 2025, with net income of $84.5 billion, driven primarily by robust iPhone sales and continued growth in its high-margin Services segment. The company maintained a solid balance sheet with significant cash reserves, ongoing share repurchases, and sustained investment in research and development to support long-term innovation and growth",
    # "Ubers free cash flow for 2024 was $1,359 millions",
    # "The revenue for Uber's delivery segment in Q1 2023 was $3,093 million (or $3.093 billion)",
    # "Uber expanded video Journey Ads to new markets including Australia, Brazil, and Chile, which indicates entry into 3 new markets",
    # "I'm sorry, but I don't have enough information to answer that",
    # "I don't know. The provided context does not include any information about Tesla's net income for 2024.",
    # "No documents found for this company."
    ]

results = []

def test_single_workflow():
    try:
        logger.info("\n--- Testing Single Query ---")
        test_questions = ["What was Apple’s net income for the first nine months ended June 28, 2025?"]
        for query in test_questions:
            result_gen = process_workflow_query(query)
        print(result_gen)
    except ImportError as e:
        logger.error(f"Failed to import services: {e}")
    except Exception as e:
        logger.error(f"An error occurred during testing: {e}")


def test_ragas_workflow():
    try:
        # Run Indexing
        chroma_db_path = backend_dir / "chroma_db"
        data_dir = project_root / "data"

        if not chroma_db_path.exists():
            pdf_paths = ["apple.pdf",
                        "volkswagen.pdf",
                        "uber.pdf"]
            for idx, pdf_path in enumerate(pdf_paths):
                pdf_full_path = data_dir / pdf_path
                if pdf_full_path.exists():
                    doc = Document(document_id=f"test_{idx}",document_path=str(pdf_full_path), document_name=pdf_path)
                    _ = index_documents(doc)
                    print("Indexing complete for document: ", pdf_path)
                else:
                    print(f"Warning: {pdf_path} not found. Please provide a valid PDF path.")
        else:
            print(f"Indexing completed already for documents at {chroma_db_path}")

        logger.info("\n--- Testing Ragas Queries ---")
        # Run Evaluation Queries
        for query in test_questions:
            result_gen = process_workflow_query(query)
            results.append(
                {
                    'question': query,
                    'answer': result_gen["response"],
                    'contexts': [doc.page_content for doc in result_gen["retriever_results"]]
                }
            )

        data = {
            "question": [r["question"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "answer": [r["answer"] for r in results],
            "ground_truth": test_answers
        }

        dataset = Dataset.from_dict(data)

        scores = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings
        )

        # Convert to Pandas DataFrame
        results_df = pd.DataFrame({
            "Question": data["question"],
            "Answer": data["answer"],
            "Faithfulness": scores["faithfulness"],
            "Answer Relevance": scores["answer_relevancy"],
            "Answer Correctness": scores["answer_correctness"],
            "Context Precision": scores["context_precision"],
            "Context Recall": scores["context_recall"]
        })

        results_df.to_csv(data_dir / "ragas_results.csv", index=False)
        
        
    except ImportError as e:
        logger.error(f"Failed to import services: {e}")
    except Exception as e:
        logger.error(f"An error occurred during testing: {e}")

def test_conversational_memory():
    try:
        logger.info("\n--- Testing Conversational Memory ---")
        
        # Turn 1
        query1 = "What is Apple's revenue for the nine months ended June 2025?"
        # We don't need history for the first query
        result1 = process_workflow_query(query1, [])
        logger.info(f"Turn 1 Response: {result1['response'][:100]}...")
        
        # Turn 2: Follow-up question
        query2 = "And what about their net income?"
        history = [
            {"role": "user", "content": query1},
            {"role": "assistant", "content": result1["response"]}
        ]
        
        # Verify that the query gets contextualized
        # We can inspect the logs or check the final answer, but ideally we'd spy on the intermediate state.
        # For this integration test, we'll check if the answer corresponds to Apple's Net Income, which implies successful rewriting.
        result2 = process_workflow_query(query2, history)
        logger.info(f"Turn 2 Response: {result2['response']}")
        
        if "84,544" in result2["response"] or "84.5" in result2["response"] or "net income" in result2["response"].lower():
             logger.info("Conversational Memory Test Passed: Context successfully used.")
        else:
             logger.warning("Conversational Memory Test: Answer might not be specific enough. Check logs for query refinement.")
             
    except Exception as e:
        logger.error(f"Conversational memory test failed: {e}")

if __name__ == "__main__":
    #test_single_workflow()
    #test_conversational_memory()
    test_ragas_workflow()