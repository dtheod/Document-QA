import yaml
import os
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from backend.indexing import get_retriever
from huggingface_hub import InferenceClient
from backend.utils import load_prompt_from_yaml, get_company_docs
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ClassificationOutput(BaseModel):
    company: str = Field(..., description="The company the user is asking about (apple, volkswagen, uber, general, unknown).")
    query_type: str = Field(..., description="The intent of the user (rag, summarize, unknown).")

class JudgeOutput(BaseModel):
    is_valid: bool = Field(..., description="True if the answer is accurate, valid and helpful, False otherwise.")
    feedback: str = Field(..., description="Detailed feedback on why the answer is valid or invalid.")

class State(TypedDict):
    query: str
    query_type: str
    query_safety_type: str
    company: str
    retriever_results: str
    response: str
    judge_result: bool
    judge_feedback: str
    chat_history: list[dict]

llm = ChatOpenAI(temperature=0, model=os.getenv("OPENAI_MODEL_NAME"))



def injection_agent(state: State) -> State:
    """
    This agent will take as input the user raw query and return SAFE OR INJECTION 
    based on the query. I added a try except block to handle the case the evaluator does
    not set the key in the .env file. In that case, it will return SAFE.
    """
    query = state["query"]
    logger.info(f"--- Injection Agent --- Query: {query}")
    try:
        client = InferenceClient(
            provider="hf-inference",
            api_key=os.getenv("HF_TOKEN"),
        )

        result = client.text_classification(
            query,
            model="protectai/deberta-v3-base-prompt-injection-v2",
        )
        dicts = {}
        for injection_type in result:
            dicts[injection_type.label] = injection_type.score

        query_type = max(dicts, key=dicts.get)
    except Exception as e:
        logger.error(f"Failed to load Huggingface model: {e}")
        query_type = "SAFE"
    
    state["query_safety_type"] = query_type
    logger.info(f"--- Injection Agent --- Safety Type: {query_type}")
    return state


def contextualize_agent(state: State) -> State:
    """
    Contextualize the query using chat history.
    """
    query = state["query"]
    chat_history = state.get("chat_history", [])
    
    if not chat_history:
        return state

    logger.info(f"--- Contextualize Agent --- Chat History: {len(chat_history)}")
    
    # Format chat history for prompt
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    # Load prompt
    contextualize_prompt = load_prompt_from_yaml(yml_type="rag_core.yml",
                                                 prompt_id="contextualize_prompt_v1")
    
    chain = contextualize_prompt | llm
    response = chain.invoke({"chat_history": history_str, "query": query})
    
    refined_query = response.content
    state["query"] = refined_query
    logger.info(f"--- Contextualize Agent --- Refined Query: {refined_query}")
    
    return state


def classification_agent(state: State) -> State:
    """
    This agent will take as input the raw query and decide if the question refers to financials or general.
    """
    query = state["query"]
    logger.info(f"--- Classification Agent --- Query: {query}")
    
    # Load prompt from YAML
    classification_prompt = load_prompt_from_yaml(yml_type="rag_core.yml",
                                                  prompt_id="classification_prompt_v1")
    
    structured_llm = llm.with_structured_output(ClassificationOutput)
    
    chain = classification_prompt | structured_llm
    response = chain.invoke({"query": query})
    
    company = response.company
    query_type = response.query_type
    
    state["company"] = company
    state["query_type"] = query_type
    logger.info(f"--- Classification Agent --- Company: {company}, Type: {query_type}")
    return state

def retriever_agent(state: State) -> State:
    
    query = state["query"]
    company = state["company"]
    query_type = state["query_type"]
    logger.info(f"--- Retriever Agent --- Query: {query}, Company: {company}, Type: {query_type}")

    retriever, _ = get_retriever()  # ParentDocumentRetriever and Vector store
    
    search_kwargs = {"k": 5}
    if company not in ["unknown", "general"]:
        search_kwargs["filter"] = {"company": company}

    # 1) Get child chunks with scores from Chroma
    results_with_scores = retriever.vectorstore.similarity_search_with_score(
        query,
        **search_kwargs,
    )

    parent_doc_ids = set()
    doc_id_to_score = {}

    for child_doc, distance in results_with_scores:
        parent_id = child_doc.metadata.get("doc_id")
        if not parent_id:
            continue

        parent_doc_ids.add(parent_id)

        # Aggregate score per parent.
        # Chroma score is usually a distance: smaller = better.
        if parent_id not in doc_id_to_score:
            doc_id_to_score[parent_id] = distance
        else:
            doc_id_to_score[parent_id] = min(doc_id_to_score[parent_id], distance)

    # 2) Fetch parent docs from docstore
    retrieved_docs = []

    if parent_doc_ids:
        parent_ids = list(parent_doc_ids)
        parent_docs = retriever.docstore.mget(parent_ids)

        for pid, p_doc in zip(parent_ids, parent_docs):
            if not p_doc:
                continue

            distance = doc_id_to_score.get(pid)

            # Optional: convert to similarity
            similarity = 1 / (1 + distance) if distance is not None else None

            p_doc.metadata["distance"] = distance
            p_doc.metadata["relevance_score"] = similarity

            retrieved_docs.append(p_doc)

    # 3) Sort by relevance (similarity desc)
    retrieved_docs.sort(
        key=lambda d: d.metadata.get("relevance_score", 0.0),
        reverse=True,
    )
        
    state["retriever_results"] = retrieved_docs
    logger.info(f"--- Retriever Agent --- Retrieved {len(retrieved_docs)} docs")

    return state


def summarisation_agent(state: State) -> State:
    """Summarisation agent using Map-Reduce for efficiency."""
    query = state["query"]
    company = state["company"]
    logger.info(f"--- Summarisation Agent --- Query: {query}, Company: {company}")

    _, vector_store = get_retriever()
    
    # Fetch all docs for the company
    company_docs = get_company_docs(vector_store, company)
    logger.info(f"--- Summarisation Agent --- Found {len(company_docs)} docs")
    
    # Load prompt from YAML
    rag_prompt = load_prompt_from_yaml(yml_type="rag_core.yml",
                                       prompt_id="summarisation_prompt_v1")
    
    # Standard chain
    chain = rag_prompt | llm
    
    # Simple strategy: If too many docs, use Map-Reduce
    # But first, satisfy the requirement: "retrieved_results in the state. maximum 5 documents"
    # We will store the first 5 docs for display purposes.
    state["retriever_results"] = company_docs[:5] if company_docs else []

    if not company_docs:
        state["response"] = "No documents found for this company."
        return state

    # Batch Aggregation (Map-Reduce)
    batch_size = 20
    if len(company_docs) > batch_size:
        logger.info("--- Summarisation Agent --- Triggering Map-Reduce")
        
        # 1. Map Step: Summarize chunks in batches
        batches = [company_docs[i:i + batch_size] for i in range(0, len(company_docs), batch_size)]
        
        # Prepare inputs for batch processing
        batch_inputs = [{"query": query, "context": b} for b in batches]
        
        # Run in parallel (threaded)
        batch_responses = chain.batch(batch_inputs)
        
        # Extract content
        intermediate_summaries = [res.content for res in batch_responses]
        combined_text = "\n\n".join(intermediate_summaries)
        
        # 2. Reduce Step: Summarize the summaries
        # We can treat the combined summaries as "context"
        final_response = chain.invoke({"query": query, "context": combined_text})
        response_content = final_response.content
        
    else:
        # standard approach for small doc counts
        response = chain.invoke({"query": query,
                                 "context": company_docs})
        response_content = response.content
    
    state["response"] = response_content
    logger.info("--- Summarisation Agent --- Response generated")
    
    return state

def qa_agent(state: State) -> State:

    """Stub for qa agent."""
    query = state["query"]
    retriever_results = state["retriever_results"]
    logger.info(f"--- QA Agent --- Query: {query}, Docs: {len(retriever_results)}")

    # Load prompt from YAML
    rag_prompt = load_prompt_from_yaml(yml_type="rag_core.yml",
                                       prompt_id="qa_agent_v1")
    
    # Prepare context from retrieved docs
    context_text = "\n\n".join([doc.page_content for doc in retriever_results])
    citations = "\n".join([f"[{doc.metadata.get('id')}]" for doc in retriever_results])

    chain = rag_prompt | llm
    response = chain.invoke({"query": query,
                             "context": context_text,
                             "citations": citations})
    
    state["response"] = response.content
    return state

def general_agent(state: State) -> State:
    """Stub for general agent."""
    query = state["query"]
    logger.info(f"--- General Agent --- Query: {query}")
    
    # Load prompt from YAML
    rag_prompt = load_prompt_from_yaml(yml_type="rag_core.yml",
                                       prompt_id="general_agent_v1")
    
    chain = rag_prompt | llm
    response = chain.invoke({"query": query})
    
    state["response"] = response.content
    state["retriever_results"] = []
    return state

def judge_agent(state: State) -> State:
    """Judge agent that validates the response."""
    query = state["query"]
    response = state["response"]
    retrieved_docs = state["retriever_results"]

    # Load prompt from YAML
    rag_prompt = load_prompt_from_yaml(yml_type = "rag_eval.yml",
                                       prompt_id = "judge_agent_v1")
    
    structured_llm = llm.with_structured_output(JudgeOutput)
    
    chain = rag_prompt | structured_llm
    result = chain.invoke({"query": query,
                             "answer": response,
                             "retrieved_docs": retrieved_docs})
    
    state["judge_result"] = result.is_valid
    state["judge_feedback"] = result.feedback
    
    logger.info(f"--- Judge Agent --- Valid: {result.is_valid}, Feedback: {result.feedback}")
    return state