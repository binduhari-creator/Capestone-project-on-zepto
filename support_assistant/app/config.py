"""
config.py
=========
Configuration module for the Zepto Support Assistant.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# MOCK_LLM toggle:
# Default (unset or "1"): fully offline, deterministic mock mode (NO API key required).
# "0": optional real-LLM extension mode.
MOCK_LLM_ENV = os.environ.get("MOCK_LLM", "1").strip().lower()
IS_MOCK_LLM = MOCK_LLM_ENV not in ("0", "false", "no")

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]
