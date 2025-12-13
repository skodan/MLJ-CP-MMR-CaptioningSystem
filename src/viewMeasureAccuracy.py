# import numpy as np
# import faiss
# from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# IMG_EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "flickr8k_train_image_embs.npy"
# TXT_EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "flickr8k_train_text_embs.npy"

# def normalize(vectors):
#     return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# image_embs = normalize(np.load(IMG_EMB_PATH))
# text_embs = normalize(np.load(TXT_EMB_PATH))

# index = faiss.IndexFlatIP(image_embs.shape[1])
# index.add(image_embs)
# top_k = 5
# scores, indices = index.search(text_embs, top_k)
# correct_image_indices = np.repeat(np.arange(len(image_embs)), 5)

# recall_at_k = (correct_image_indices[:, None] == indices).any(axis=1).mean()
# print(f"Recall@{top_k}: {recall_at_k:.4f}")

from huggingface_hub import hf_hub_download
from pathlib import Path
import os

# --- Configuration ---
REPO_ID = "jxie/flickr8k"
REPO_TYPE = "dataset"
# The name of the file you are looking for inside the repository
FILENAME = "flickr8k_train.csv" 
# The local folder in your project where the file should go
LOCAL_DOWNLOAD_DIR = Path("D:\Learnings\AIML\CapstoneProject\workspace\MLJ-CP-MMR-CaptioningSystem\data\captions")

# 1. Create the local directory if it doesn't exist
os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)

# 2. Download the file
local_file_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    repo_type=REPO_TYPE,
    local_dir=LOCAL_DOWNLOAD_DIR,
    # Setting this to False ensures the actual file is copied to local_dir
    local_dir_use_symlinks=False 
)

print(f"File '{FILENAME}' downloaded successfully to: {local_file_path}")