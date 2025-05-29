import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.document_chunker import DocumentChunker
from src.rag.embedder import HybridIndexer
from src.rag.agent import ConversationalRetrievalAgent
from config import CLEANED_DATA_DIR, CHROMA_DB_DIR, BM25_DIR
from langchain_core.runnables import RunnableConfig

# docs = DocumentChunker(CLEANED_DATA_DIR)
# docs.load_documents()
# docs.split_documents()
def test_agent(config: RunnableConfig):
    
    indexer = HybridIndexer(persist_directory=CHROMA_DB_DIR, bm25_directory=BM25_DIR)
    indexer.create()

    agent = ConversationalRetrievalAgent(indexer, temperature=0.7)

    return agent.app

if __name__ == "__main__":
    # docs = DocumentChunker(CLEANED_DATA_DIR)
    # docs.load_documents()
    # docs.split_documents()

    indexer = HybridIndexer(persist_directory=CHROMA_DB_DIR, bm25_directory=BM25_DIR)
    indexer.create()
    
    k_docs = indexer.hybrid_search("địa điểm vui chơi", k=5)
    print(k_docs)