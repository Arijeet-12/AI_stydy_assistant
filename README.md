# AI Study Assistant

A college-ready AI Study Assistant that lets students upload PDFs/notes, search their content, ask questions, generate summaries, flashcards and MCQs, and track study progress.

## Features
- PDF and TXT upload
- Text extraction with PyMuPDF
- Local TF-IDF retrieval (no API key required)
- RAG-style question answering with source snippets
- Optional OpenAI-compatible LLM integration
- Automatic chapter/document summaries
- Flashcard generation
- MCQ generation
- Quiz mode with scoring
- Study progress dashboard
- SQLite persistence
- Clean responsive frontend

## Run
### 1. Backend
```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Frontend
The frontend is plain HTML/CSS/JS, so no Node.js is required.
Open `frontend/index.html` in a browser after starting the backend. If your browser blocks local API requests, use a simple server:
```bash
cd frontend
python -m http.server 5500
```
Then visit `http://localhost:5500`.

## Optional AI model
The project works without an external AI key. To enable generated answers, create `backend/.env`:
```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_key
LLM_MODEL=gpt-4o-mini
```
The backend uses an OpenAI-compatible `/chat/completions` endpoint. You can point it at another compatible provider/local server.

## Project structure
```text
ai-study-assistant/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── README.md
```
