"""
ingestion.py
============
Document ingestion and local vector database indexing using SentenceTransformers and ChromaDB.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import DOCS_DIR, CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME

# Global cache for embedding model and Chroma client
_EMBEDDING_MODEL = None
_CHROMA_CLIENT = None


def get_embedding_model():
    """
    Load and cache local SentenceTransformer model.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        print(f"Loading local embedding model: {EMBEDDING_MODEL_NAME}...")
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


def get_chroma_client():
    """
    Initialize and cache persistent ChromaDB client.
    """
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        _CHROMA_CLIENT = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    return _CHROMA_CLIENT


def ingest_documents(force_reindex=False):
    """
    Reads all 8 policy documents, generates local embeddings, and indexes them in ChromaDB.
    """
    client = get_chroma_client()
    model = get_embedding_model()

    if force_reindex:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    existing_count = collection.count()
    if existing_count >= 8 and not force_reindex:
        print(f"ChromaDB collection '{COLLECTION_NAME}' already contains {existing_count} documents.")
        return collection

    doc_files = sorted(list(DOCS_DIR.glob("doc_*.txt")))
    if not doc_files:
        raise FileNotFoundError(f"No policy documents found in: {DOCS_DIR}")

    ids = []
    documents = []
    metadatas = []

    for file_path in doc_files:
        doc_id = file_path.stem  # e.g., "doc_01"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Derive title from content or document id
        title = doc_id
        if " — " in content:
            title = content.split(" — ")[0]

        ids.append(doc_id)
        documents.append(content)
        metadatas.append({"source": doc_id, "title": title, "filename": file_path.name})

    # Generate embeddings locally
    print(f"Generating local embeddings for {len(documents)} policy documents...")
    embeddings = model.encode(documents, convert_to_numpy=True).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Successfully ingested and indexed {collection.count()} policy chunks into ChromaDB.")
    return collection


if __name__ == "__main__":
    coll = ingest_documents(force_reindex=True)
    print(f"Collection count: {coll.count()}")
