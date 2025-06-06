import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

# Project directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
CLEANED_DATA_DIR = os.path.join(DATA_DIR, "processed")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")
BM25_DIR = os.path.join(DATA_DIR, "bm25")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

# Document chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
GLOB_PATTERN = "./**/*.md"

# Embedding model
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"

