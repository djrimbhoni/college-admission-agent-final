# 🎓 College Admission Agent (RAG Based)

A Retrieval-Augmented Generation (RAG) assistant designed to streamline the college admission inquiry process. Built for the **AICTE-2026 Problem Statement No. 4**, this solution uses **IBM Granite** models via IBM Watsonx to retrieve precise, grounded answers on eligibility, tuition fees, deadlines, and campus facilities directly from institutional handbooks.

---

## 🏗 System Architecture

The project ingests document data into a vector database and exposes an interactive chat interface backed by IBM Watsonx LLM and embedding models.

┌───────────────────┐      Ingestion      ┌────────────┐
│ data/handbook.txt │ ──────────────────> │ ingest.py  │
└───────────────────┘                     └─────┬──────┘
│ Embeddings: ibm/granite-embedding-278m-multilingual
▼
┌──────────────────┐
│ Chroma Vector DB │ (admission_db)
└────────┬─────────┘
│ Top 4 Chunks
▼
┌──────────────────┐  POST /ask   ┌────────────────┐  Query Context  ┌───────────────┐
│ Browser Chat UI  │ <──────────> │ FastAPI main.py│ <─────────────> │  rag_agent.py │
└──────────────────┘              └────────────────┘                 └───────┬───────┘
│ LLM: ibm/granite-4-h-small
▼
┌───────────────┐
│  IBM Watsonx  │
└───────────────┘


---

## ✨ Features

- **Grounded Information Retrieval:** Strictly answers queries using local document contexts (e.g., `data/handbook.txt`) to eliminate hallucinations.
- **Powered by IBM Granite:** Uses `ibm/granite-embedding-278m-multilingual` for document vectorization and `ibm/granite-4-h-small` for natural language generation.
- **FastAPI Core:** Lightweight back-end delivering endpoints for UI rendering and query responses.
- **Embedded Web UI:** Responsive, zero-dependency HTML/JS chat interface built into FastAPI with quick-prompt suggestions.

---

## 🛠 Tech Stack & Requirements

- **Framework:** Python 3.9+, FastAPI, Uvicorn
- **Orchestration:** LangChain, LangChain IBM, LangChain Chroma
- **Vector Database:** ChromaDB
- **Models (IBM Watsonx):** 
  - LLM: `ibm/granite-4-h-small`
  - Embeddings: `ibm/granite-embedding-278m-multilingual`

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/college-admission-rag-agent.git](https://github.com/your-username/college-admission-rag-agent.git)
cd college-admission-rag-agent
2. Set Up Virtual Environment & Install Dependencies
Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
3. Configure Environment Variables
Create a .env file in the root directory:

Code snippet
WATSONX_APIKEY="your-ibm-cloud-apikey"
WATSONX_PROJECT_ID="your-watsonx-project-id"
WATSONX_URL="[https://us-south.ml.cloud.ibm.com](https://us-south.ml.cloud.ibm.com)"
4. Prepare Document & Ingest Vectors
Place your college handbook text file inside the data/ directory (e.g., data/handbook.txt), then run the ingestion script:

Bash
python ingest.py
This splits the text into chunks, generates embeddings using IBM Granite, and saves the vector store locally in admission_db/.

5. Run the Application
Start the FastAPI application with Uvicorn:

Bash
uvicorn main:app --reload
Open your browser and navigate to:

http://localhost:8000
📁 Repository Structure
.
├── admission_db/          # Local Chroma vector database directory (generated after ingest)
├── data/
│   └── handbook.txt       # College admission policies, fee structure, and FAQs
├── architecture_blueprint.jpg # System visual architecture diagram
├── ingest.py              # Ingestion pipeline: chunks text and builds Chroma vector store
├── main.py                # FastAPI server & interactive single-page web UI
├── rag_agent.py           # LangChain RAG retrieval pipeline & Watsonx integration
├── requirements.txt       # Python project dependencies
└── .env                   # Environment credentials (Git ignored)
📜 License
Developed under the IBM SkillsBuild for University Engagements (AICTE-2026) initiative.