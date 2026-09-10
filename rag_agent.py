"""
rag_agent.py
------------
RAG retrieval pipeline powered by IBM Granite via WatsonxLLM.

Usage (standalone test):
    python rag_agent.py

Or import and call answer() from main.py:
    from rag_agent import answer
    response = answer("What is the tuition fee?")
"""

import os
from dotenv import load_dotenv
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

CHROMA_DIR = "admission_db"
EMBED_MODEL = "ibm/granite-embedding-278m-multilingual"
LLM_MODEL = "ibm/granite-4-h-small"

# ---------------------------------------------------------------------------
# Prompt — instructs the model to answer strictly from retrieved context
# ---------------------------------------------------------------------------
_PROMPT_TEMPLATE = """You are a helpful college admissions assistant.
Answer the student's question using ONLY the information provided in the
context below. If the answer is not found in the context, respond with:
"I'm sorry, that information is not available in the college handbook."

Context:
{context}

Student question: {question}

Answer:"""

ADMISSION_PROMPT = PromptTemplate(
    template=_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _build_chain():
    """Build the LCEL RAG chain."""
    embeddings = WatsonxEmbeddings(
        model_id=EMBED_MODEL,
        url=WATSONX_URL,
        apikey=WATSONX_APIKEY,
        project_id=WATSONX_PROJECT_ID,
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    llm = WatsonxLLM(
        model_id=LLM_MODEL,
        url=WATSONX_URL,
        apikey=WATSONX_APIKEY,
        project_id=WATSONX_PROJECT_ID,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 512,
            "min_new_tokens": 10,
            "temperature": 0.0,
        },
    )

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | ADMISSION_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# Module-level chain (lazy-initialised on first call)
_chain = None


def answer(query: str) -> str:
    """Return a grounded answer for *query* using the RAG pipeline."""
    global _chain
    if _chain is None:
        _chain = _build_chain()
    return _chain.invoke(query).strip()


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_questions = [
        "What is the minimum percentage required for admission?",
        "What is the tuition fee per year?",
        "What are the hostel charges for an AC room?",
        "When is the Round 1 application deadline?",
        "What JEE Main percentile is required?",
    ]
    for q in test_questions:
        print(f"\nQ: {q}")
        print(f"A: {answer(q)}")
