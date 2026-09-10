"""
main.py
-------
FastAPI application for the College Admission RAG Agent.

Endpoints:
  GET  /      — Serves the chat-box UI (HTML + JS, no external deps)
  POST /ask   — Accepts {"query": "..."} and returns {"answer": "..."}

Start the server:
    uvicorn main:app --reload
Then open http://localhost:8000 in your browser.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag_agent import answer as rag_answer

app = FastAPI(title="College Admission RAG Agent")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str


class AnswerResponse(BaseModel):
    answer: str


# ---------------------------------------------------------------------------
# Chat UI — served at GET /
# ---------------------------------------------------------------------------
_CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>College Admission Assistant</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
      background: #f0f4f8;
      display: flex;
      justify-content: center;
      align-items: flex-start;
      min-height: 100vh;
      padding: 32px 16px;
    }

    .card {
      background: #ffffff;
      border-radius: 12px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.10);
      width: 100%;
      max-width: 720px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    header {
      background: #1d3461;
      color: #ffffff;
      padding: 20px 24px;
    }
    header h1 { font-size: 1.25rem; font-weight: 600; }
    header p  { font-size: 0.82rem; opacity: 0.75; margin-top: 4px; }

    #chat-window {
      flex: 1;
      padding: 20px 24px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
      min-height: 380px;
      max-height: 520px;
    }

    .bubble {
      max-width: 82%;
      padding: 10px 14px;
      border-radius: 10px;
      font-size: 0.9rem;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .bubble.user {
      align-self: flex-end;
      background: #1d3461;
      color: #ffffff;
      border-bottom-right-radius: 2px;
    }
    .bubble.bot {
      align-self: flex-start;
      background: #eef2ff;
      color: #1f2328;
      border-bottom-left-radius: 2px;
    }
    .bubble.thinking {
      align-self: flex-start;
      background: #eef2ff;
      color: #6b7280;
      font-style: italic;
      border-bottom-left-radius: 2px;
    }

    .input-row {
      display: flex;
      gap: 10px;
      padding: 16px 24px;
      border-top: 1px solid #e5e7eb;
      background: #fafafa;
    }

    #query-input {
      flex: 1;
      padding: 10px 14px;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      font-size: 0.9rem;
      outline: none;
      transition: border-color 0.2s;
    }
    #query-input:focus { border-color: #1d3461; }

    #send-btn {
      padding: 10px 20px;
      background: #1d3461;
      color: #ffffff;
      border: none;
      border-radius: 8px;
      font-size: 0.9rem;
      cursor: pointer;
      transition: background 0.2s;
    }
    #send-btn:hover   { background: #274880; }
    #send-btn:disabled { background: #9ca3af; cursor: default; }

    .suggestions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 0 24px 16px;
    }
    .suggestions button {
      background: #eef2ff;
      border: 1px solid #c7d2fe;
      border-radius: 20px;
      padding: 5px 12px;
      font-size: 0.78rem;
      color: #1d3461;
      cursor: pointer;
      transition: background 0.15s;
    }
    .suggestions button:hover { background: #c7d2fe; }

    footer {
      text-align: center;
      padding: 10px;
      font-size: 0.75rem;
      color: #9ca3af;
      border-top: 1px solid #e5e7eb;
    }
  </style>
</head>
<body>
<div class="card">
  <header>
    <h1>🎓 College Admission Assistant</h1>
    <p>Powered by IBM Granite · Ask about fees, eligibility, deadlines &amp; more</p>
  </header>

  <div id="chat-window"></div>

  <div class="suggestions">
    <button onclick="fillQuery('What is the minimum eligibility for admission?')">Eligibility criteria</button>
    <button onclick="fillQuery('What is the annual tuition fee?')">Tuition fee</button>
    <button onclick="fillQuery('What are the hostel charges for an AC room?')">Hostel charges</button>
    <button onclick="fillQuery('What JEE Main percentile is required?')">JEE cutoff</button>
    <button onclick="fillQuery('When is the Round 1 application deadline?')">Application deadline</button>
  </div>

  <div class="input-row">
    <input id="query-input" type="text" placeholder="Type your question here…" autocomplete="off" />
    <button id="send-btn" onclick="sendQuery()">Send</button>
  </div>

  <footer>Made with IBM Bob &nbsp;|&nbsp; IBM Granite &nbsp;|&nbsp; IBM Cloud Lite</footer>
</div>

<script>
  const chatWindow = document.getElementById('chat-window');
  const input      = document.getElementById('query-input');
  const sendBtn    = document.getElementById('send-btn');

  function appendBubble(text, role) {
    const div = document.createElement('div');
    div.className = 'bubble ' + role;
    div.textContent = text;
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return div;
  }

  function fillQuery(text) {
    input.value = text;
    input.focus();
  }

  async function sendQuery() {
    const query = input.value.trim();
    if (!query) return;

    appendBubble(query, 'user');
    input.value = '';
    sendBtn.disabled = true;

    const thinking = appendBubble('Thinking…', 'thinking');

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      const data = await res.json();
      thinking.remove();
      appendBubble(data.answer || 'No response received.', 'bot');
    } catch (err) {
      thinking.remove();
      appendBubble('Error: Could not reach the server. Please try again.', 'bot');
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') sendQuery();
  });

  // Welcome message
  appendBubble('Hello! I can answer questions about admissions, fees, eligibility, and deadlines. How can I help you today?', 'bot');
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the chat-box UI."""
    return HTMLResponse(content=_CHAT_HTML)


@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QueryRequest):
    """Run the RAG pipeline and return a grounded answer."""
    response = rag_answer(request.query)
    return AnswerResponse(answer=response)
