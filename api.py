# api.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from typing import List
from pydantic import BaseModel

from model_registry import get_model
from models.resnet_lstm_attention.schemas import CaptionResult, ImageResult, TextQuery

app = FastAPI(title="Multimodal Retrieval & Captioning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    model_name: str
    top_k: int = 5

#@app.post("/caption", response_model=CaptionResult)
@app.post("/caption")
async def caption_image(model_name: str = Form(...), file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    model = get_model(model_name)
    caption = model.generate_caption(image)
    return {"caption": caption}

#@app.post("/search/text2img", response_model=List[ImageResult])
@app.post("/search/text2img")
async def text_to_image(model_name: str = Form(...), query: str = Form(...), top_k: int = Form(5)):
    model = get_model(model_name)
    results = model.text_to_image(query, top_k)
    return results

@app.post("/search/img2text")
async def image_to_text(model_name: str = Form(...), file: UploadFile = File(...), top_k: int = Form(5)):
    image = Image.open(file.file).convert("RGB")
    model = get_model(model_name)
    results = model.image_to_text(image, top_k)
    return results

#@app.post("/search/img2img", response_model=List[ImageResult])
@app.post("/search/img2img")
async def image_to_image(model_name: str = Form(...), file: UploadFile = File(...), top_k: int = Form(5)):
    image = Image.open(file.file).convert("RGB")
    model = get_model(model_name)
    results = model.image_to_image(image, top_k)
    return results

@app.get("/health")
def health_check():
    return {"status": "healthy"}