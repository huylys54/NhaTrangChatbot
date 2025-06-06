from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.api.models import ChatRequest, ChatResponse, ImageData
from src.rag.agent import ConversationalRetrievalAgent
from src.rag.embedder import HybridIndexer
from src.rag.document_chunker import DocumentChunker
import json
import os
from config import CLEANED_DATA_DIR, CHROMA_DB_DIR, BM25_DIR, IMAGES_DIR


router = APIRouter()

image_metadata_file = os.path.join(IMAGES_DIR, 'metadata.json')
# # Load RAG components on startup
# chunker = DocumentChunker(dir_path=CLEANED_DATA_DIR)
# chunker.load_documents()
# chunker.split_documents()

indexer = HybridIndexer(persist_directory=CHROMA_DB_DIR, bm25_directory=BM25_DIR, image_metadata_file=image_metadata_file)
indexer.create()

agent = ConversationalRetrievalAgent(indexer=indexer, streaming=True)


@router.post("/chat")
async def ask_chatbot(req: ChatRequest):
    """Stream the chatbot's response as chunks with automatic image detection."""
    def generate():
        # Run the workflow to get context and determine if images should be included
            history = agent.memory.load_memory_variables({})["chat_history"]
            init_state = {
                "query": req.query,
                "language": req.language if req.language else "",
                "context": "",
                "history": history,
                "intent": "",
                "include_images": False,
                "images": [],
                "error": ""
            }
            
            # Run workflow to get context and images
            final_state = agent.app.invoke(init_state)
            
            # Check if images were found
            has_images = final_state.get("include_images", False) and final_state.get("images", [])
            
            # Start response with image metadata if available
            if has_images:
                images_data = final_state["images"]
                images_json = json.dumps({
                    "type": "images",
                    "data": images_data
                })
                yield f"__IMAGES__{images_json}__IMAGES__\n"
            
            # Stream text response
            full_response = ""
            for chunk in agent.stream_response(
                final_state["query"], 
                final_state["language"], 
                final_state["context"], 
                final_state["history"]
            ):
                full_response += chunk
                yield chunk
            
            # Save to memory
            agent.memory.save_context(
                inputs={"human": req.query}, 
                outputs={"ai": full_response}
            )

    return StreamingResponse(generate(), media_type="text/plain")


@router.post("/chat-complete", response_model=ChatResponse)
async def ask_chatbot_complete(req: ChatRequest):
    """Get complete response with automatic image detection (non-streaming)."""
    result = agent.ask(req.query, req.language)
    
    # Extract images if available
    images = []
    has_images = False
    if result.get("include_images", False) and result.get("images", []):
        has_images = True
        for img_data in result["images"]:
            images.append(ImageData(**img_data))
    
    return ChatResponse(
        response=result["response"],
        images=images,
        has_images=has_images
    )



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