from src.rag.document_chunker import DocumentChunker
from src.rag.embedder import HybridIndexer
import argparse

from config import CHROMA_DB_DIR, BM25_DIR, CLEANED_DATA_DIR

def main(args):
    # Initialize DocumentChunker
    chunker = DocumentChunker(
        dir_path=args.dir_path,
        chunk_size=1000,
        chunk_overlap=50
    )
    
    chunker.load_documents()
    chunker.split_documents()


    # Create or load hybrid indexer
    indexer = HybridIndexer(
        all_sections=chunker.all_sections,
        persist_directory=args.persist_directory,
        bm25_directory=args.bm25_directory
    )
    
    indexer.create()
    print("Hybrid indexer created or loaded successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create or load hybrid indexer.")
    parser.add_argument("--persist_directory", type=str, default=CHROMA_DB_DIR, help="Directory to persist Chroma index.")
    parser.add_argument("--bm25_directory", type=str, default=BM25_DIR, help="Directory to persist BM25 index.")
    parser.add_argument("--dir_path", type=str, default=CLEANED_DATA_DIR, help="Directory containing markdown documents to chunk.")
    args = parser.parse_args()
    main(args)