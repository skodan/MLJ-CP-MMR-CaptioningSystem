import numpy as np
import faiss

embeddings = np.load('data/embeddings/flickr8k_train_image_embs.npy')
print(f'Embeddings shape: {embeddings.shape}')
print("First embedding vector:", embeddings[0][:5])

def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

image_embs = normalize(np.load("flickr8k_train_image_embs.npy"))
text_embs = normalize(np.load("flickr8k_train_text_embs.npy"))

index = faiss.IndexFlatIP(image_embs.shape[1])
index.add(image_embs)
top_k = 5
scores, indices = index.search(text_embs, top_k)
# Assume captions are ordered: 5 captions per image, in order
# So caption 0–4 → image 0, 5–9 → image 1, etc.
correct_image_indices = np.repeat(np.arange(len(image_embs)), 5)

# Check if correct index is in top K results
recall_at_k = (correct_image_indices[:, None] == indices).any(axis=1).mean()
print(f"Recall@{top_k}: {recall_at_k:.4f}")
