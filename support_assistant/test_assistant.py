"""
test_assistant.py
=================
Verification script for the Zepto Support Assistant.
Tests intent routing, ChromaDB semantic retrieval, deterministic canned response generation,
and Pydantic schema validation.
"""

import sys
import os
import json

# Ensure support_assistant directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from app.main import app
from app.ingestion import ingest_documents
from app.config import IS_MOCK_LLM


def run_tests():
    print("=" * 80)
    print("ZEPTO SUPPORT ASSISTANT - VERIFICATION TEST SUITE")
    print(f"Mode: {'MOCK_LLM (Deterministic Offline Baseline)' if IS_MOCK_LLM else 'REAL_LLM'}")
    print("=" * 80)

    # 1. Verify ingestion
    print("\n[STEP 1] Verifying Document Ingestion in ChromaDB...")
    coll = ingest_documents()
    doc_count = coll.count()
    print(f"-> ChromaDB collection verified: {doc_count} documents indexed.")
    assert doc_count >= 8, f"Expected at least 8 documents, found {doc_count}"

    # 2. Test Client
    client = TestClient(app)

    test_queries = [
        {
            "type": "Policy Query (Delivery)",
            "query": "What is the delivery policy and standard delivery fee for Zepto?",
            "expected_intent": "policy_question",
            "expected_source": "doc_01"
        },
        {
            "type": "Policy Query (Returns & Refunds)",
            "query": "How do I return a damaged grocery item and get a refund?",
            "expected_intent": "policy_question",
            "expected_source": "doc_02"
        },
        {
            "type": "Policy Query (Membership Tiers)",
            "query": "What are the membership tiers and Zepto Pass perks?",
            "expected_intent": "policy_question",
            "expected_source": "doc_03"
        },
        {
            "type": "Policy Query (Cancellation)",
            "query": "How can I cancel my order before it gets packed?",
            "expected_intent": "policy_question",
            "expected_source": "doc_05"
        },
        {
            "type": "General / Out-of-Scope Query",
            "query": "What is the capital city of Australia?",
            "expected_intent": "general_question",
            "expected_source": None
        }
    ]

    print("\n[STEP 2] Running Endpoint Query Tests (/ask)...")
    for t in test_queries:
        print(f"\n--- Testing {t['type']} ---")
        print(f"Query: \"{t['query']}\"")
        resp = client.post("/ask", json={"query": t["query"]})
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()

        print(f"Status: {resp.status_code}")
        print("Raw JSON Response:")
        print(json.dumps(data, indent=2))

        # Validations
        assert "answer" in data and len(data["answer"]) > 0
        assert "confidence" in data and 0.0 <= data["confidence"] <= 1.0
        assert "sources" in data and isinstance(data["sources"], list)

        if t["expected_source"]:
            assert t["expected_source"] in data["sources"], f"Expected {t['expected_source']} in sources, got {data['sources']}"
            assert data["answer"].startswith("Based on the retrieved context:")
        else:
            assert len(data["sources"]) == 0
            assert data["answer"] == "I can only answer questions about Zepto policies right now."

    print("\n" + "=" * 80)
    print("ALL SUPPORT ASSISTANT VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_tests()
