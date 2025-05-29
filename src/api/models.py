from pydantic import BaseModel

class ChatRequest(BaseModel):
    """
    Represents a chat request with query, user_id and language.
    """
    
    query: str
    user_id: str = "default"
    
    
class SearchRequest(BaseModel):
    """
    Represents a search request with query, user_id and language.
    """
    
    query: str
    
class MetadataResponse(BaseModel):
    """
    Represents metadata response containing information about the index."""
    
    documents: int
    images: int
    last_updated: str