"""
All configuration lives here. Every other module imports from this file
instead of reading environment variables itself.

Environment variables -- Setup

Test -- python -c "from config import settings; print(settings.ROOT, settings.OPENAI_MODEL)"
-- Should Return the environment variable -- gpt-4.1-mini -- 
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains config/
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- Databases ---
DATABASE_URL = os.environ["DATABASE_URL"]
READONLY_DATABASE_URL = os.environ["READONLY_DATABASE_URL"]

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

# --- Vector database ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DOCS_COLLECTION = "olist_docs"
EXAMPLES_COLLECTION = "olist_examples"

# --- LLM ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- Paths ---
RAW_DIR = ROOT / "data" / "raw"
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"

# --- Safety / behaviour ---
MAX_ROWS = 1000          # row limit injected into every query
MAX_RETRIES = 3          # self-correction attempts

ALLOWED_TABLES = {
    "customers",
    "sellers",
    "products",
    "category_translation",
    "orders",
    "order_items",
    "order_payments",
    "order_reviews",
    "geolocation",
}