from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.api.models import ChatRequest
from src.rag.agent import ConversationalRetrievalAgent
from src.rag.embedder import HybridIndexer
from src.rag.document_chunker import DocumentChunker
from config import CLEANED_DATA_DIR, CHROMA_DB_DIR, BM25_DIR


router = APIRouter()

# # Load RAG components on startup
# chunker = DocumentChunker(dir_path=CLEANED_DATA_DIR)
# chunker.load_documents()
# chunker.split_documents()

indexer = HybridIndexer(persist_directory=CHROMA_DB_DIR, bm25_directory=BM25_DIR)
indexer.create()

agent = ConversationalRetrievalAgent(indexer=indexer, streaming=True)


@router.post("/chat")
async def ask_chatbot(req: ChatRequest):
    """
    Stream the chatbot's response as chunks.
    """
    def generate():
        # Yield chunks from the agent's ask method
        for chunk in agent.ask_streaming(req.query):
            yield chunk
    
    return StreamingResponse(generate(), media_type="text/plain")

@router.get("/history")
def get_history():
    history = agent.memory.load_memory_variables({})
    return history

@router.delete("/history")
def clear_history():
    agent.memory.clear()
    return {"message": "History cleared."}

# @router.get("/documents/search")
# def document_search(query: str):
#     docs = indexer.hybrid_search(query, k=10)
#     results = [d.page_content for d in docs]
#     return {"results": results}

# @router.get("/metadata")
# def get_metadata():
#     meta = {
#         "documents": len(chunker.all_sections),
#         "images": 0,
#         "last_updated": "2025-05-28"
#     }
#     return meta