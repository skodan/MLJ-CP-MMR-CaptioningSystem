import pandas as pd
import pickle
import numpy as np
import faiss
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = PROJECT_ROOT / "data" / "captions" / "flickr8k_train.csv"
VOCAB_OUTPUT = PROJECT_ROOT / "data" / "processed" / "vocab.pkl"
IMG_EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "flickr8k_train_image_embs.npy"
TXT_EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "flickr8k_train_text_embs.npy"

def build_vocab(csv_path, output_path):
    df = pd.read_csv(csv_path)
    captions = df["caption"].tolist()

    all_words = []
    for caption in captions:
        words = caption.lower().strip().split()
        all_words.extend(words)

    unique_words = sorted(set(all_words))

    # Add special tokens
    unique_words = ["<BOS>", "<EOS>", "<PAD>", "<UNK>"] + unique_words

    word2int = {word: i for i, word in enumerate(unique_words)}
    int2word = {i: word for word, i in word2int.items()}

    # Save to pickle file
    with open(output_path, "wb") as f:
        pickle.dump((word2int, int2word), f)

    print(f"Vocabulary saved to {output_path}")
    print(f"Total tokens: {len(word2int)}")


def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def evaluate_recall_at_k(IMG_EMB_PATH, TXT_EMB_PATH, top_k=5):
    image_embs = normalize(np.load(IMG_EMB_PATH))
    text_embs = normalize(np.load(TXT_EMB_PATH))

    index = faiss.IndexFlatIP(image_embs.shape[1])
    index.add(image_embs)
    
    scores, indices = index.search(text_embs, top_k)
    correct_image_indices = np.repeat(np.arange(len(image_embs)), top_k)

    recall_at_k = (correct_image_indices[:, None] == indices).any(axis=1).mean()
    print(f"Recall@{top_k}: {recall_at_k:.4f}")


# Example usage
if __name__ == "__main__":
    build_vocab(
        csv_path= CSV_FILE,
        output_path= VOCAB_OUTPUT
    )
