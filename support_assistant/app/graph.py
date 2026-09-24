"""
graph.py
========
LangGraph StateGraph orchestration for Zepto Support Assistant.
Implements intent classification, semantic retrieval, canned deterministic mock generation,
and conditional routing.
"""

import os
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from app.config import IS_MOCK_LLM, POLICY_KEYWORDS
from app.retrieval import retrieve_top_k_chunks


class AssistantState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    confidence: float


# Structured prompt template specification for real-LLM mode (Task 2)
STRUCTURED_PROMPT_TEMPLATE = """
Role: You are Zepto's Official Customer Support AI Assistant.
Context:
{context}

Task: Answer the user question accurately based ONLY on the provided policy context above.
Format: Return a clean, concise paragraph without extraneous filler.
Length: Between 2 and 4 sentences.

Negative Constraint: Do NOT answer using information not present in the provided context. If the policy context does not contain the answer, state that the information is unavailable.

Few-Shot Example:
User: "How much is standard delivery?"
Context: "Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee."
Answer: "Standard delivery is free for all orders above INR 149. Orders below this amount incur a flat delivery charge of INR 25."

User Query: "{query}"
"""


def classify_intent_node(state: AssistantState) -> Dict[str, Any]:
    """
    Classify the incoming query as 'policy_question' or 'general_question'.
    Uses deterministic keyword heuristic in MOCK_LLM mode.
    """
    query_lower = state["query"].lower().strip()

    # Keyword heuristic required by specification
    is_policy = any(kw in query_lower for kw in POLICY_KEYWORDS)
    intent = "policy_question" if is_policy else "general_question"

    # Optional real-LLM extension branch (when MOCK_LLM=0)
    if not IS_MOCK_LLM:
        # If real LLM integration is configured via API key
        pass

    return {"intent": intent}


def retrieve_and_answer_node(state: AssistantState) -> Dict[str, Any]:
    """
    Retrieves top-3 relevant chunks from ChromaDB via cosine similarity.
    In MOCK_LLM mode, returns a deterministic canned templated answer.
    """
    query = state["query"]
    # Real ChromaDB semantic retrieval runs in both mock and real modes
    chunks = retrieve_top_k_chunks(query, top_k=3)

    if not chunks:
        return {
            "retrieved_chunks": [],
            "answer": "No relevant policy documents could be retrieved.",
            "sources": [],
            "confidence": 0.0
        }

    sources = [c["id"] for c in chunks]

    if IS_MOCK_LLM:
        # Canned deterministic response required by specification:
        # "Based on the retrieved context: {top_chunk_snippet}" (first ~200 characters)
        top_snippet = chunks[0]["text"][:200]
        answer = f"Based on the retrieved context: {top_snippet}"
        confidence = 1.0
    else:
        # Optional real-LLM extension logic
        top_snippet = chunks[0]["text"][:200]
        answer = f"Based on the retrieved context: {top_snippet}"
        confidence = 1.0

    return {
        "retrieved_chunks": chunks,
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }


def direct_answer_node(state: AssistantState) -> Dict[str, Any]:
    """
    Handles general/out-of-scope questions directly without retrieval.
    """
    if IS_MOCK_LLM:
        answer = "I can only answer questions about Zepto policies right now."
        confidence = 1.0
    else:
        answer = "I can only answer questions about Zepto policies right now."
        confidence = 1.0

    return {
        "retrieved_chunks": [],
        "answer": answer,
        "sources": [],
        "confidence": confidence
    }


def route_intent(state: AssistantState) -> str:
    """
    Conditional edge routing function.
    """
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"


def build_graph():
    """
    Constructs and compiles the LangGraph StateGraph.
    """
    workflow = StateGraph(AssistantState)

    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer_node)
    workflow.add_node("direct_answer", direct_answer_node)

    workflow.set_entry_point("classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer"
        }
    )

    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)

    app = workflow.compile()
    return app


# Cached compiled graph instance
_GRAPH_APP = None


def get_graph_app():
    global _GRAPH_APP
    if _GRAPH_APP is None:
        _GRAPH_APP = build_graph()
    return _GRAPH_APP
