import os

from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "tractor-manual-hierarchical")
PINECONE_DENSE_INDEX_NAME = os.getenv("PINECONE_DENSE_INDEX_NAME", "rag-documents-ubuntu-dense")
HF_TOKEN = os.getenv("HF_TOKEN", None)
BGE_M3_MODEL = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")
BGE_RERANKER_MODEL = os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

CHUNKS_DIR = "output/parsed/chunks"
INDEX_JSONL = "output/parsed/chunks/index.jsonl"
EMBEDDING_DIM = 1024
METRIC = "dotproduct"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/indexing.log")
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() in ("1", "true", "yes")
QUERY_LOG_FILE = os.getenv("QUERY_LOG_FILE", "logs/query.log")

# Noise filter settings
NOISE_FILTER_ENABLED = os.getenv("NOISE_FILTER_ENABLED", "true").lower() in ("1", "true", "yes")
NOISE_CHAR_THRESHOLD = float(os.getenv("NOISE_CHAR_THRESHOLD", "0.8"))
NOISE_MIN_LINE_LENGTH = int(os.getenv("NOISE_MIN_LINE_LENGTH", "5"))
NOISE_CHARS = set(os.getenv("NOISE_CHARS", "0.").replace(" ", "").split(","))
# Add bullet chars not splittable by comma
NOISE_CHARS.update("•|")
# Additional patterns that are pure noise
NOISE_LINE_PATTERNS = [
    "0 0 0",  # Repeated zeros with spaces
    ". . .",  # Repeated dots with spaces
    "• • •",  # Repeated bullets with spaces
    "|• |•",  # Table noise
]

# Token limits for chunk splitting
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "2048"))
RECURSIVE_CHUNK_SIZE = int(os.getenv("RECURSIVE_CHUNK_SIZE", "1024"))
RECURSIVE_OVERLAP = int(os.getenv("RECURSIVE_OVERLAP", "100"))

# Processed chunks output settings
WRITE_PROCESSED_CHUNKS = os.getenv("WRITE_PROCESSED_CHUNKS", "true").lower() in ("1", "true", "yes")
PROCESSED_CHUNKS_DIR = os.getenv("PROCESSED_CHUNKS_DIR", "output/parsed/chunks_processed")