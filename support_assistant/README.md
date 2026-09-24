# Module 3: Grounded GenAI Support Assistant

## 1. Module Purpose
Module 3 implements an intelligent, policy-grounded Customer Support AI Assistant for the **Zepto Data & AI Platform Capstone**. The assistant uses a local Retrieval-Augmented Generation (RAG) pipeline orchestrated via **LangGraph**, **ChromaDB**, and **SentenceTransformers** to classify user query intent, semantically retrieve relevant sections of Zepto\'s internal policy corpus, and return validated, structured answers.

The graded baseline is **100% offline, deterministic, and keyless**, powered by a robust `MOCK_LLM` mode that executes semantic retrieval from ChromaDB and returns structured Pydantic responses with zero external API dependencies.

---

## 2. Architecture & Data Flow

```
User Query (POST /ask)
         │
         ▼
┌───────────────────────────────┐
│     LangGraph StateGraph      │
│                               │
│      [classify_intent]        │
│    (Keyword/LLM Routing)      │
└──────────────┬────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
[policy_question]   [general_question]
      │                 │
      ▼                 ▼
┌─────────────────────────┐  ┌─────────────────────────────────┐
│  retrieve_and_answer    │  │          direct_answer          │
│                         │  │                                 │
│ 1. Local Query Embedding│  │ Returns fixed canned response:  │
│    (all-MiniLM-L6-v2)   │  │ "I can only answer questions    │
│ 2. ChromaDB Top-3 Cosine│  │  about Zepto policies..."       │
│    Similarity Retrieval │  │ Sources: []                     │
│ 3. Grounded Canned /    │  │ Confidence: 1.0                 │
│    Structured Answer    │  └────────────────┬────────────────┘
│ Sources: [doc_ids]      │                   │
│ Confidence: 1.0         │                   │
└──────────────┬──────────┘                   │
               │                              │
               └──────────────┬───────────────┘
                              ▼
               ┌──────────────────────────────┐
               │    Pydantic AskResponse      │
               │ (answer, sources, confidence)│
               └──────────────────────────────┘
```

### Stage Walkthrough
1. **Ingestion & Chunking**: Reads all 8 official Zepto policy documents from `docs/` and indexes them into persistent storage.
2. **Local Embedding**: Embeds chunks locally with the open-source `sentence-transformers` library using `all-MiniLM-L6-v2` (384-dimensional dense vectors; no network calls or paid APIs required).
3. **ChromaDB Vector Store**: Manages the local persistent vector collection `zepto_policies` with cosine similarity space (`hnsw:space: cosine`).
4. **Intent Classification**: Evaluates incoming queries using deterministic keyword heuristics (`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`). Routes policy-related queries to `retrieve_and_answer` and general questions to `direct_answer`.
5. **Semantic Retrieval**: For policy questions, queries ChromaDB to retrieve the **Top-3 most similar chunks** based on vector cosine similarity.
6. **Answer Generation**:
   - **`MOCK_LLM=1` / Unset (Graded Baseline)**: Generates a deterministic answer grounded in the top retrieved snippet: `f"Based on the retrieved context: {top_chunk_snippet}"`.
   - **`MOCK_LLM=0` (Optional Extension)**: Generates an answer using a structured prompt with few-shot examples and schema validation retries.
7. **Structured Pydantic Response**: Enforces schema validation returning `answer` (str), `sources` (list of document IDs), and `confidence` (float 0.0–1.0).

---

## 3. Official Policy Document Corpus (8 Documents)

The corpus contains the 8 policy documents defined in the official specification:

| Document ID | Title | Summary / Core Policy |
| :--- | :--- | :--- |
| `doc_01` | **Delivery Policy** | 10–30 min delivery, free on orders > INR 149 (INR 25 fee below), priority delivery +INR 15. |
| `doc_02` | **Returns & Refunds** | Perishables returnable in 24h; non-perishables in 7 days; 3–5 day refund or instant wallet credit. |
| `doc_03` | **Membership Tiers** | Basic (Free), Zepto Pass (INR 49/mo, free delivery, 5% off), Zepto Pass+ (INR 99/mo, priority delivery, 10% off). |
| `doc_04` | **Order Tracking** | Live rider map; contact support if no movement for > 20 mins past ETA. |
| `doc_05` | **Order Cancellation** | Free cancellation before "Packed" status (~first 2 mins). Auto-refunded if delivery fails. |
| `doc_06` | **Damaged/Missing Items**| Report in 24h via "Report an Issue"; free replacement/refund (photos required for orders > INR 1000). |
| `doc_07` | **Gift Cards** | Denominations of INR 100/250/500/1000; valid for 1 year; combinable with one payment method. |
| `doc_08` | **Customer Support Hours**| 24/7 in-app chat (< 2 min response); email response within 24h on business days; no phone support. |

---

## 4. `MOCK_LLM` Mode & Configuration

The application is controlled by the `MOCK_LLM` environment variable:
- **`MOCK_LLM=1` or Unset (Default Graded Baseline)**:
  - Runs completely **offline** with **zero external API calls**.
  - Intent classification uses deterministic keyword matching.
  - ChromaDB semantic retrieval runs locally for real.
  - Generates deterministic, grounded responses with `confidence = 1.0`.
  - Requires **NO API key** and **NO network access**.
- **`MOCK_LLM=0` (Optional Real LLM Extension)**:
  - Connects to an external LLM (e.g. Groq free tier) to synthesize grounded narrative answers using the structured prompt template.

### Structured Prompt Specification (for Real LLM Mode)
```
Role: You are Zepto's Official Customer Support AI Assistant.
Context: {retrieved_policy_context}
Task: Answer the user question accurately based ONLY on the provided policy context above.
Format: Return a clean, concise paragraph without extraneous filler.
Length: Between 2 and 4 sentences.
Negative Constraint: Do NOT answer using information not present in the provided context.
Few-Shot Example:
  User: "How much is standard delivery?"
  Context: "Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee."
  Answer: "Standard delivery is free for all orders above INR 149. Orders below this amount incur a flat delivery charge of INR 25."
```

---

## 5. Live API Demonstrations & Captured JSON Outputs

The following raw JSON responses were captured from actual queries against the local `/ask` endpoint:

### Example 1: Policy Question (Delivery Policy)
**Request**: `POST /ask`
```json
{
  "query": "What is the delivery policy and standard delivery fee for Zepto?"
}
```
**Actual Response (`HTTP 200 OK`)**:
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": [
    "doc_01",
    "doc_03",
    "doc_05"
  ],
  "confidence": 1.0
}
```

---

### Example 2: General / Out-of-Scope Question
**Request**: `POST /ask`
```json
{
  "query": "What is the capital city of Australia?"
}
```
**Actual Response (`HTTP 200 OK`)**:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

---

### Example 3: Policy Question (Returns & Refunds)
**Request**: `POST /ask`
```json
{
  "query": "How do I return a damaged grocery item and get a refund?"
}
```
**Actual Response (`HTTP 200 OK`)**:
```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unop",
  "sources": [
    "doc_02",
    "doc_06",
    "doc_05"
  ],
  "confidence": 1.0
}
```

---

## 6. Local Setup & Execution

### 1. Install Dependencies
```bash
pip install -r support_assistant/requirements.txt
```

### 2. Run Automated Verification Test Suite
```bash
python support_assistant/test_assistant.py
```

### 3. Start the FastAPI Service
```bash
python -m uvicorn app.main:app --app-dir support_assistant --host 0.0.0.0 --port 7860
```
Interactive API documentation will be available at `http://localhost:7860/docs`.

---

## 7. Docker Deployment

### Dockerfile Specification
```dockerfile
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MOCK_LLM=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docs ./docs
COPY app ./app

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Build and Run with Docker
```bash
# Build Docker Image
docker build -t zepto-support-assistant:latest ./support_assistant

# Run Container
docker run -d -p 7860:7860 --name zepto-assistant zepto-support-assistant:latest

# Test /ask endpoint
curl -X POST http://localhost:7860/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the delivery fee?"}'
```
