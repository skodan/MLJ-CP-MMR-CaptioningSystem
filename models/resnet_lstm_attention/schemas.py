from pydantic import BaseModel
from typing import List

class TextQuery(BaseModel):
    query: str
    top_k: int = 5

class ImageResult(BaseModel):
    image_path: str
    score: float

class CaptionResult(BaseModel):
    caption: str
