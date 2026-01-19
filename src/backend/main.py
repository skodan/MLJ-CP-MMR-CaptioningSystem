from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import json
import torch
import os

from loader import load_captioning_model                 # M2 caption loader
from clip_loader import load_clip_model, CLIPRetrievalModel                  # CLIP-style retrieval loader
from retrieval import RetrievalService                   # FAISS retrieval logic
from schemas import CaptionResult, ImageResult, TextQuery                             # API schemas


MODEL_DIR = os.getenv("MODEL_DIR", "models")
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", "embeddings")
VOCAB_CAP_PATH = os.getenv("VOCAB_CAP_PATH", "vocab")
CONFIGS_DIR = os.getenv("CONFIGS_DIR", "configs")

# ---------------------------------------------
# FastAPI app
# ---------------------------------------------
app = FastAPI(title="Multimodal Retrieval & Captioning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cpu")

# ---------------------------------------------
# Startup: load ALL models once
# ---------------------------------------------
@app.on_event("startup")
def load_models():
    global caption_bundle, retrieval_service

    # ===============================
    # 1. Load Captioning Model (M2)
    # ===============================
    try:
        caption_bundle = load_captioning_model(
            model_path=os.path.join(MODEL_DIR, "caption_model.pth"),
            vocab_path=os.path.join(VOCAB_CAP_PATH, "vocab.pkl"),
            config_path=os.path.join(CONFIGS_DIR, "caption_config.json"),
            device=DEVICE
        )
    except Exception as e:
        print("Error loading captioning model:", e)
        raise RuntimeError("Failed to load captioning model")

    # caption_bundle contains:
    # {
    #   "encoder", "decoder", "vocab",
    #   "inv_vocab", "max_len"
    # }

    # ===============================
    # 2. Load Retrieval Model (CLIP-style)
    # ===============================
    with open(os.path.join(CONFIGS_DIR, "preprocess_config.json"), "r") as f:
        preprocess_cfg = json.load(f)

    clip_model = load_clip_model(
        model_path=os.path.join(MODEL_DIR, "flickr8k_retrieval_model.pth"),
        vocab=caption_bundle["vocab"],
        device=DEVICE
    )

    retrieval_service = RetrievalService(
        clip_model=clip_model,
        image_index_path=os.path.join(EMBEDDINGS_DIR, "image_embeddings.faiss"),
        text_index_path=os.path.join(EMBEDDINGS_DIR, "text_embeddings.faiss"),
        image_map_path=os.path.join(EMBEDDINGS_DIR, "image_id_map.pkl"),
        text_map_path=os.path.join(EMBEDDINGS_DIR, "text_id_map.pkl"),
        preprocess=preprocess_cfg
        #device=DEVICE
    )

    print("✅ Captioning and Retrieval models loaded successfully")


# ---------------------------------------------
# Endpoints
# ---------------------------------------------

@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    """
    Image → Caption (M2: ResNet + LSTM + Attention)
    """
    image = Image.open(file.file).convert("RGB")

    encoder = caption_bundle["encoder"]
    decoder = caption_bundle["decoder"]
    vocab = caption_bundle["vocab"]
    inv_vocab = caption_bundle["inv_vocab"]
    max_len = caption_bundle["max_len"]
    transform = caption_bundle["transform"]

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = encoder(image_tensor)
        tokens = decoder.generate(
            features,
            vocab=vocab,
            inv_vocab=inv_vocab,
            max_len=max_len
        )

    return {"caption": " ".join(tokens)}


@app.post("/search/text2img")
async def text_to_image(query: TextQuery):
    """
    Text → Image (CLIP-style retrieval)
    """
    results = retrieval_service.text_to_image(
        query.query,
        top_k=query.top_k
    )
    return results


@app.post("/search/img2text")
async def image_to_text(file: UploadFile = File(...)):
    """
    Image → Text (CLIP-style retrieval)
    """
    image = Image.open(file.file).convert("RGB")
    captions = retrieval_service.image_to_text(image)
    return captions


@app.post("/search/img2img", response_model=List[ImageResult])
async def image_to_image(file: UploadFile = File(...)):
    """
    Image → Image (retrieve similar images)
    """
    image = Image.open(file.file).convert("RGB")
    
    # Preprocess and encode the uploaded image
    image_tensor = retrieval_service.image_transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        emb = retrieval_service.clip_model.encode_image(image_tensor).cpu().numpy()
    
    emb = retrieval_service._normalize(emb)
    
    # Search in image embeddings index
    scores, idxs = retrieval_service.image_index.search(emb, 5)
    
    results = [
        {
            "image_path": str(retrieval_service.image_id_map[i]),  # as string
            "score": float(scores[0][j])
        }
        for j, i in enumerate(idxs[0])
    ]
    
    return results

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": True}