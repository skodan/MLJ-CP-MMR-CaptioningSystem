import faiss
import pickle
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class RetrievalService:
    def __init__(self, clip_model, image_index_path, text_index_path,
                 image_map_path, text_map_path, preprocess):

        self.device = torch.device("cpu")
        self.clip_model = clip_model

        self.image_index = faiss.read_index(image_index_path)
        self.text_index = faiss.read_index(text_index_path)

        with open(image_map_path, "rb") as f:
            self.image_id_map = pickle.load(f)

        with open(text_map_path, "rb") as f:
            self.text_id_map = pickle.load(f)

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=preprocess["mean"],
                std=preprocess["std"]
            )
        ])

    def _normalize(self, x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def text_to_image(self, text, top_k=5):
        with torch.no_grad():
            emb = self.clip_model.encode_text(text).cpu().numpy()
        emb = self._normalize(emb)

        scores, idxs = self.image_index.search(emb, top_k)
        return [
            {
                "image_path": self.image_id_map[i],
                "score": float(scores[0][j])
            }
            for j, i in enumerate(idxs[0])
        ]

    def image_to_text(self, image: Image.Image, top_k=5):
        image = self.image_transform(image).unsqueeze(0)
        with torch.no_grad():
            emb = self.clip_model.encode_image(image).cpu().numpy()
        emb = self._normalize(emb)

        scores, idxs = self.text_index.search(emb, top_k)
        results = [self.text_id_map[i] for i in idxs[0]]
        print(f"DEBUG: Returning results: {results}")
        return results
