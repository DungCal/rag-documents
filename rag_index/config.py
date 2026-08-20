import os

from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "tractor-manual-hierarchical")
HF_TOKEN = os.getenv("HF_TOKEN", None)
BGE_M3_MODEL = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")

CHUNKS_DIR = "output/parsed/chunks"
INDEX_JSONL = "output/parsed/chunks/index.jsonl"
EMBEDDING_DIM = 1024
METRIC = "cosine"