import os
from dotenv import load_dotenv

load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HF_API_URL = os.getenv("HF_API_URL", "")
HF_API_PROXY = os.getenv("HF_API_PROXY", "")
BM25_THRESHOLD = float(os.getenv("BM25_THRESHOLD", "25"))
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.12"))
TOP_N_FOR_LLM = int(os.getenv("TOP_N_FOR_LLM", "10"))
USE_DYNAMIC_THRESHOLDS = os.getenv("USE_DYNAMIC_THRESHOLDS", "true").lower() == "true"
STAGE1_KEEP_PERCENT = float(os.getenv("STAGE1_KEEP_PERCENT", "60"))
STAGE2_KEEP_COUNT = int(os.getenv("STAGE2_KEEP_COUNT", "15"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
USE_TRANSFORMER_EMBEDDINGS = os.getenv("USE_TRANSFORMER_EMBEDDINGS", "false").lower() == "true"
WEIGHT_BM25 = 0.20
WEIGHT_SEMANTIC = 0.30
WEIGHT_LLM = 0.50
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(10 * 1024 * 1024)))
MAX_RESUMES = int(os.getenv("MAX_RESUMES", "50"))
