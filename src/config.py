# --- Data Source Configuration ---
PARQUET_FILE_PATH = "./data/raw/edgar_2020.parquet"
CIK_FILTER = "5272"  # Set to None to process all CIKs in the file

# --- Evaluation Data Configuration ---
GROUND_TRUTH_CSV_PATH = "./data/ground_truth.csv"
RESULTS_CSV_PATH = "./results_comparison.csv"  # The final output file
TARGET_VARIABLES = ["Share-based compensation", "Net investment income", "Product Line"]

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dim

# --- Text Splitting Configuration ---
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 300

# --- Vector Store Configuration ---
CHROMA_PERSIST_DIR = "vector_db"

# --- LLM Configuration ---
GEMINI_MODEL_NAME = "gemini-2.5-pro"
