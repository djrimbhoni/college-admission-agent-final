"""
ingest.py
---------
Reads data/handbook.txt, splits it into chunks, embeds each chunk
using IBM Granite embeddings via WatsonxEmbeddings, and persists the
resulting vector store to a local Chroma database folder (admission_db).

Run once (or re-run whenever the handbook is updated):
    python ingest.py
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxEmbeddings
from langchain_chroma import Chroma

load_dotenv()

WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

HANDBOOK_PATH = "data/handbook.txt"
CHROMA_DIR = "admission_db"
EMBED_MODEL = "ibm/granite-embedding-278m-multilingual"


def main():
    # 1. Load the handbook
    print(f"Loading document: {HANDBOOK_PATH}")
    loader = TextLoader(HANDBOOK_PATH, encoding="utf-8")
    documents = loader.load()

    # 2. Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from the handbook.")

    # 3. Initialise WatsonxEmbeddings
    embeddings = WatsonxEmbeddings(
        model_id=EMBED_MODEL,
        url=WATSONX_URL,
        apikey=WATSONX_APIKEY,
        project_id=WATSONX_PROJECT_ID,
    )

    # 4. Build and persist the Chroma vector store
    print(f"Embedding chunks and writing vector store to '{CHROMA_DIR}' …")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print("Ingestion complete. Vector store is ready.")


if __name__ == "__main__":
    main()
