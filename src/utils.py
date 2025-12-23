import pandas as pd
import pickle
import numpy as np
from pathlib import Path
import re
from collections import Counter
from datasets import load_dataset
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from jiwer import wer
from huggingface_hub import snapshot_download
from pathlib import Path
import os
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import ast

try:
    import faiss
except ImportError:
    faiss = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = PROJECT_ROOT / "data" / "captions" / "flickr8k_train.csv"
VOCAB_OUTPUT = PROJECT_ROOT / "data" / "captions" / "vocab.pkl"
IMG_EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "flickr8k_train_image_embs.npy"
TXT_EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "flickr8k_train_text_embs.npy"

IMAGE_DIR = PROJECT_ROOT / "data" / "30k-images"
ANNOTATION_CSV = PROJECT_ROOT / "data" / "annotations_30k.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"

# --- Configuration ---
REPO_ID = "nlphuji/flickr30k"
REPO_TYPE = "dataset"
LOCAL_DOWNLOAD_DIR = Path("./data/30k") 


# Function to download the entire dataset snapshot from Hugging Face
def download_hf_dataset(repo_id, repo_type, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    print(f"Starting download to: {local_dir.resolve()}")

    local_path = snapshot_download(
        repo_id=repo_id, 
        repo_type=repo_type,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )

    print(f"\n--- Download Complete ---")
    print(f"Entire dataset snapshot is saved in: {local_path}")


def generate_flickr30k_clip_embeddings(
    image_dir,
    annotation_csv,
    output_dir,
    batch_size=64
):
    """
    Generate CLIP image & text embeddings for Flickr30k
    using OFFICIAL train/val/test splits from annotations.

    Expected CSV columns:
    ['raw', 'sentids', 'split', 'filename', 'img_id']
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CLIP] Using device: {device}")

    # Load CLIP
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(device)
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    model.eval()

    # Load annotations
    df = pd.read_csv(annotation_csv)

    required_cols = {"raw", "filename", "split"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Expected columns {required_cols}, found {df.columns}"
        )

    print("[CLIP] Using official Flickr30k splits")

    # Collect data by split
    data_by_split = {"train": [], "val": [], "test": []}

    for _, row in df.iterrows():
        split = row["split"]
        img_name = row["filename"]

        if split not in data_by_split:
            continue

        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue

        # 'raw' is a string representation of a Python list
        captions = ast.literal_eval(row["raw"])

        for caption in captions:
            data_by_split[split].append({
                "img_id": img_name,
                "image_path": img_path,
                "caption": caption
            })

    for s in data_by_split:
        print(f"[CLIP] {s}: {len(data_by_split[s])} captions")

    # Generate embeddings per split
    for split_name, split_data in data_by_split.items():
        print(f"\n[CLIP] Processing {split_name} split")

        split_dir = Path(output_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        img_embs, txt_embs, metadata = [], [], []

        for i in tqdm(range(0, len(split_data), batch_size)):
            batch = split_data[i:i + batch_size]

            images = [
                Image.open(d["image_path"]).convert("RGB")
                for d in batch
            ]
            texts = [d["caption"] for d in batch]

            inputs = processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            image_embeds = outputs.image_embeds
            text_embeds  = outputs.text_embeds

            # Normalize for cosine similarity
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds  = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            img_embs.append(image_embeds.cpu())
            txt_embs.append(text_embeds.cpu())
            metadata.extend(batch)

        np.save(
            split_dir / "30k_hf_image_embeddings.npy",
            torch.cat(img_embs).numpy()
        )
        np.save(
            split_dir / "30k_hf_text_embeddings.npy",
            torch.cat(txt_embs).numpy()
        )
        np.save(
            split_dir / "metadata.npy",
            np.array(metadata, dtype=object)
        )

        print(f"[CLIP] Saved {split_name} embeddings → {split_dir}")



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


def tokenize(caption):
    caption = caption.lower().strip()
    caption = re.sub(r"[^\w\s]", "", caption)  # remove punctuation
    return caption.split()


def build_vocab_from_hf_dataset(train_ds, caption_field="caption_0"):
    token_counter = Counter()
    for example in train_ds:
        tokens = tokenize(example[caption_field])
        token_counter.update(tokens)

    special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
    unique_tokens = special_tokens + sorted(token_counter.keys())

    word2int = {word: idx for idx, word in enumerate(unique_tokens)}
    int2word = {idx: word for word, idx in word2int.items()}

    vocab = {
        "word2int": word2int,
        "int2word": int2word,
        "specials": special_tokens
    }

    with open(VOCAB_OUTPUT, "wb") as f:
        pickle.dump(vocab, f)

    print(f"Vocabulary size: {len(word2int)}")
    return vocab


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


def generate_caption(model,image,word2int,int2word,max_len=20,device=None):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode image
        features = model.encoder(image)
        features = features.flatten(1)
        img_emb = model.img_fc(features)
        img_emb = img_emb.unsqueeze(1)

        caption = [word2int["<sos>"]]
        hidden = None

        for _ in range(max_len):
            caption_tensor = torch.tensor(caption).unsqueeze(0).to(device)
            embeddings = model.embedding(caption_tensor)

            lstm_input = torch.cat([img_emb, embeddings], dim=1)
            lstm_out, hidden = model.lstm(lstm_input, hidden)

            output = model.fc(lstm_out[:, -1, :])
            predicted = output.argmax(dim=-1).item()

            if int2word[predicted] == "<eos>":
                break

            caption.append(predicted)

    return " ".join([int2word[idx] for idx in caption[1:]])


def compute_metrics(reference, hypothesis):
    reference = reference.lower().split()
    hypothesis = hypothesis.lower().split()

    smoothie = SmoothingFunction().method4

    bleu1 = sentence_bleu(
        [reference],
        hypothesis,
        weights=(1, 0, 0, 0),
        smoothing_function=smoothie
    )

    bleu4 = sentence_bleu(
        [reference],
        hypothesis,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie
    )

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = rouge.score(" ".join(reference), " ".join(hypothesis))["rougeL"].fmeasure

    wer_score = wer(" ".join(reference), " ".join(hypothesis))

    return bleu1, bleu4, rouge_l, wer_score


# Example usage
if __name__ == "__main__":
    # build_vocab(
    #     csv_path= CSV_FILE,
    #     output_path= VOCAB_OUTPUT
    # )

    # ds = load_dataset("jxie/flickr8k")
    # vocab = build_vocab_from_hf_dataset(ds["train"])

    # download_hf_dataset(
    #     repo_id=REPO_ID,
    #     repo_type=REPO_TYPE,
    #     local_dir=LOCAL_DOWNLOAD_DIR
    # )
    generate_flickr30k_clip_embeddings(
        image_dir=str(IMAGE_DIR),
        annotation_csv=str(ANNOTATION_CSV),
        output_dir=str(OUTPUT_DIR),
        batch_size=64
    )
