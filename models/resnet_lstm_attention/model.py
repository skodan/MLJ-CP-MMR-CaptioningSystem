import os
import json
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
from typing import List, Dict, Any

from .loader import load_captioning_model
from .retrieval import RetrievalService
from .clip_loader import load_clip_model
from .captioning import CaptioningService  # Not directly used, but for reference
from utils.interfaces import UnifiedModelInterface  # Adjust path if needed

class ResNetLSTMAttentionModel(UnifiedModelInterface):
    def __init__(self):
        self.caption_bundle = None
        self.retrieval_service = None
        self.device = torch.device("cpu")
        #self.model_repo = "skodan/resnet-lstm-attention-weights"

    def load(self) -> None:
        if self.caption_bundle is not None and self.retrieval_service is not None:
            return
        
        MODEL_REPO = "skodan/resnet-lstm-attention-weights"

        files_to_download = [
                "caption_model.pth",
                "flickr8k_retrieval_model.pth",
                "image_embeddings.faiss",
                "text_embeddings.faiss",
                "image_id_map.pkl",
                "text_id_map.pkl",
                "vocab.pkl"
            ]

        downloaded_paths = {}
        for fname in files_to_download:
            try:
                path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=fname,
                    repo_type="model",
                )
                downloaded_paths[fname] = path
            except Exception as e:
                raise RuntimeError(f"Failed to download {fname} from {MODEL_REPO}: {e}")

        # Download large files from HF Hub
        caption_pth = downloaded_paths["caption_model.pth"]
        retrieval_pth = downloaded_paths["flickr8k_retrieval_model.pth"]
        image_index_faiss = downloaded_paths["image_embeddings.faiss"]
        text_index_faiss = downloaded_paths["text_embeddings.faiss"]
        image_map_pkl = downloaded_paths["image_id_map.pkl"]
        text_map_pkl = downloaded_paths["text_id_map.pkl"]
        vocab_pkl = downloaded_paths["vocab.pkl"]  

        # Load configs (assume small, committed to repo)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # go up to project root
        config_path = os.path.join(base_dir, "configs", "caption_config.json")
        preprocess_cfg_path = os.path.join(base_dir, "configs", "preprocess_config.json")

        with open(config_path, "r") as f:
            caption_config = json.load(f)

        with open(preprocess_cfg_path, "r") as f:
            preprocess_cfg = json.load(f)

        # Load captioning
        self.caption_bundle = load_captioning_model(
            model_path=caption_pth,
            vocab_path=vocab_pkl,
            config_path=config_path,
            device=self.device
        )

        # Load retrieval
        clip_model = load_clip_model(
            model_path=retrieval_pth,
            vocab=self.caption_bundle["vocab"],
            device=self.device
        )

        self.retrieval_service = RetrievalService(
            clip_model=clip_model,
            image_index_path=image_index_faiss,
            text_index_path=text_index_faiss,
            image_map_path=image_map_pkl,
            text_map_path=text_map_pkl,
            preprocess=preprocess_cfg
        )

        print("Model components loaded successfully.")

    @torch.no_grad()
    def generate_caption(self, image: Image.Image) -> str:
        encoder = self.caption_bundle["encoder"]
        decoder = self.caption_bundle["decoder"]
        vocab = self.caption_bundle["vocab"]
        inv_vocab = self.caption_bundle["inv_vocab"]
        max_len = self.caption_bundle["max_len"]
        transform = self.caption_bundle["transform"]

        image_tensor = transform(image).unsqueeze(0).to(self.device)
        features = encoder(image_tensor)
        tokens = decoder.generate(
            features,
            vocab=vocab,
            inv_vocab=inv_vocab,
            max_len=max_len
        )
        return " ".join(tokens)

    def text_to_image(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.retrieval_service.text_to_image(text, top_k)

    def image_to_text(self, image: Image.Image, top_k: int = 5) -> List[str]:
        return self.retrieval_service.image_to_text(image, top_k)

    def image_to_image(self, image: Image.Image, top_k: int = 5) -> List[Dict[str, Any]]:
        image_tensor = self.retrieval_service.image_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.retrieval_service.clip_model.encode_image(image_tensor).cpu().numpy()
        emb = self.retrieval_service._normalize(emb)
        scores, idxs = self.retrieval_service.image_index.search(emb, top_k)
        return [
            {"image_path": self.retrieval_service.image_id_map[i], "score": float(scores[0][j])}
            for j, i in enumerate(idxs[0])
        ]