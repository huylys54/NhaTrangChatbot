from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str
    language: Optional[str] = None

class ImageData(BaseModel):
    image_id: str
    path: str
    caption: str
    tags: List[str]
    relevance_score: Optional[float] = None

class ChatResponse(BaseModel):
    response: str
    images: Optional[List[ImageData]] = None
    has_images: bool = False