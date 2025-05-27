import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.document_chunker import DocumentChunker
from src.rag.embedder import HybridIndexer
from src.rag.generator import ConversationalRetrievalAgent
from config import CLEANED_DATA_DIR, CHROMA_DB_DIR, TFIDF_DIR

# docs = DocumentChunker(CLEANED_DATA_DIR)
# docs.load_documents()
# docs.split_documents()

indexer = HybridIndexer(persist_directory=CHROMA_DB_DIR, tfidf_directory=TFIDF_DIR)
indexer.create()


agent = ConversationalRetrievalAgent(indexer, temperature=0.7)





while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chatbot.")
        break
    answer = agent.ask(user_input)
    print("GoNhaTrang:", answer)