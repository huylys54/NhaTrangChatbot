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
TFIDF_DIR = os.path.join(DATA_DIR, "tfidf")

# Document chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
GLOB_PATTERN = "./**/*.md"

# Embedding model
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


# Crawler settings
CRAWLER_URLS = [
    'https://nhatrang.khanhhoa.gov.vn/gioi-thieu-chung',
    'https://nhatrang.khanhhoa.gov.vn/di-tich-lich-su-danh-lam-thang-canh',
    'https://nhatrang.khanhhoa.gov.vn/tour-du-lich',
    'https://nhatrang.khanhhoa.gov.vn/am-thuc',
    'https://nhatrang.khanhhoa.gov.vn/luu-tru',
    'https://nhatrang.khanhhoa.gov.vn/thong-tin-huu-ich',
    'https://nhatrang.khanhhoa.gov.vn/so-dien-thoai-can-thiet'
]

