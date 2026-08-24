import os

from dotenv import load_dotenv

load_dotenv()

# Pinecone Configuration
PINECONE_HOST = os.getenv("PINECONE_HOST", "")  # e.g. "http://localhost:5080" for Pinecone Local
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "") or ("pclocal" if os.getenv("PINECONE_HOST") else "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "tractor-manual-hierarchical")
PINECONE_DENSE_INDEX_NAME = os.getenv("PINECONE_DENSE_INDEX_NAME", "rag-documents-ubuntu-dense")
HF_TOKEN = os.getenv("HF_TOKEN", None)
BGE_M3_MODEL = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")

EMBEDDER_TYPE = os.getenv("EMBEDDER_TYPE", "local")
LOCAL_BGE_M3_PATH = os.getenv("LOCAL_BGE_M3_PATH", "bge-m3/bge-m3")
EMBEDDER_DEVICE = os.getenv("EMBEDDER_DEVICE", "cuda")

CHUNKS_DIR = "output/parsed/chunks"
INDEX_JSONL = "output/parsed/chunks/index.jsonl"
EMBEDDING_DIM = 1024
METRIC = os.getenv("PINECONE_METRIC", "cosine")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/indexing.log")
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() in ("1", "true", "yes")