"""
main.py
=======
FastAPI application serving the Zepto Support Assistant API.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import AskRequest, AskResponse
from app.ingestion import ingest_documents
from app.graph import get_graph_app
from app.config import IS_MOCK_LLM


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ingest policy documents and build LangGraph app
    print(f"Starting Zepto Support Assistant (MOCK_LLM mode: {IS_MOCK_LLM})...")
    ingest_documents()
    get_graph_app()
    yield
    print("Shutting down Zepto Support Assistant...")


app = FastAPI(
    title="Zepto Customer Support AI Assistant",
    description="RAG-powered conversational assistant for Zepto customer policies.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "mock_mode": IS_MOCK_LLM}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Process user query through LangGraph pipeline:
    1. Intent classification (policy_question vs general_question)
    2. Context retrieval & answer generation
    3. Structured Pydantic response validation
    """
    try:
        graph = get_graph_app()
        initial_state = {
            "query": request.query,
            "intent": "",
            "retrieved_chunks": [],
            "answer": "",
            "sources": [],
            "confidence": 0.0
        }

        final_state = graph.invoke(initial_state)

        response = AskResponse(
            answer=final_state["answer"],
            sources=final_state["sources"],
            confidence=final_state["confidence"]
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=False)
