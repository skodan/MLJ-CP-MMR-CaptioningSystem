import pandas as pd
import pickle
import numpy as np
from pathlib import Path
import re
from collections import Counter
from datasets import load_dataset
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge_score import rouge_scorer
# from jiwer import wer
# from huggingface_hub import snapshot_download
import os
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import ast
import matplotlib.pyplot as plt
import textwrap
from src.dataloader_hf import Flickr30kImageDataset, Flickr30kEvalDataset, eval_collate_fn
from torch.utils.data import DataLoader

try:
    import faiss
except ImportError:
    faiss = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "30k-images"
ANNOTATION_CSV = PROJECT_ROOT / "data" / "annotations_30k.csv"
CSV_FILE = PROJECT_ROOT / "data" / "captions" / "flickr8k_train.csv"

VOCAB_OUTPUT = PROJECT_ROOT / "data" / "processed" / "vocab_8k.pkl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"

IMG_EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "30k_test" / "30k_hf_image_embeddings.npy"
TXT_EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "30k_test" / "30k_hf_text_embeddings.npy"
CAPTION_COUNTS_PATH = PROJECT_ROOT / "data" / "embeddings" / "train" / "train_hf-8k_caption_counts.npy"
META_PATH = PROJECT_ROOT / "data" / "embeddings" / "30k_test" / "metadata.npy"

REPO_ID = "nlphuji/flickr30k"
REPO_TYPE = "dataset"
LOCAL_DOWNLOAD_DIR = Path("./data/30k")

# load embeddings and caption counts
# image_embs = np.load(IMG_EMB_PATH)
# text_embs = np.load(TXT_EMB_PATH)
#cnt = np.load(CAPTION_COUNTS_PATH)

# image ids and text ids for HF 8k Flickr dataset
#image_ids = list(range(len(image_embs)))

# text_ids = []
# for img_id, c in enumerate(cnt):
#     text_ids.extend([img_id] * c)

# image ids and text ids for HF 30k Flickr dataset
# metadata = np.load(META_PATH, allow_pickle=True)
# img_ids = [m["img_id"] for m in metadata]

# Load models


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


def generate_retrieval_embeddings(image_encoder,text_encoder,df,vocab,image_root,device,batch_size=64):
    image_encoder.eval()
    text_encoder.eval()

    # -------- IMAGE EMBEDDINGS --------
    img_ds = Flickr30kImageDataset(df, image_root)
    img_loader = DataLoader(img_ds, batch_size=batch_size, shuffle=False)

    all_img_embs = []
    all_img_ids = []

    img_counter = 0  # <-- IMPORTANT

    with torch.no_grad():
        for images in img_loader:
            images = images.to(device)

            out = image_encoder(images)
            img_emb = out[0] if isinstance(out, tuple) else out

            all_img_embs.append(img_emb.cpu())

            # Correct image ids
            bsz = images.size(0)
            all_img_ids.extend(range(img_counter, img_counter + bsz))
            img_counter += bsz

    # with torch.no_grad():
    #     for images in img_loader:
    #         images = images.to(device)
    #         out = image_encoder(images)

    #         # Handle both normal and DataParallel outputs
    #         if isinstance(out, tuple):
    #             img_emb = out[0]
    #         else:
    #             img_emb = out

    #         all_img_embs.append(img_emb.cpu())
            # emb = image_encoder(images)
            # all_img_embs.append(emb.cpu())

    all_img_embs = torch.cat(all_img_embs).numpy()

    # -------- TEXT EMBEDDINGS --------
    txt_ds = Flickr30kEvalDataset(df, vocab, image_root)
    txt_loader = DataLoader(
        txt_ds, batch_size=batch_size,
        shuffle=False, collate_fn=eval_collate_fn
    )

    all_txt_embs = []
    all_txt_ids = []

    with torch.no_grad():
        for captions, img_ids, lengths in txt_loader:
            captions = captions.to(device)
            emb = text_encoder(captions, lengths)
            all_txt_embs.append(emb.cpu())
            all_txt_ids.extend(img_ids)

    all_txt_embs = torch.cat(all_txt_embs).numpy()
    all_txt_ids = np.array(all_txt_ids)

    return all_img_embs, all_txt_embs, all_txt_ids, np.array(all_img_ids)


def generate_flickr30k_clip_embeddings(image_dir,annotation_csv,output_dir,batch_size=64):
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


def build_vocab_from_flickr30k_annotations(captions_file,train_split_file,min_freq=2):
    token_counter = Counter()

    # Read train image filenames
    with open(train_split_file, "r") as f:
        train_images = set(line.strip() for line in f)

    # Read captions
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            img_caption, caption = line.strip().split("\t")
            img_name = img_caption.split("#")[0]

            if img_name in train_images:
                tokens = tokenize(caption)
                token_counter.update(tokens)

    # Apply min_freq
    tokens = [tok for tok, freq in token_counter.items() if freq >= min_freq]

    special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
    vocab_tokens = special_tokens + sorted(tokens)

    word2int = {w: i for i, w in enumerate(vocab_tokens)}
    int2word = {i: w for w, i in word2int.items()}

    vocab = {
        "word2int": word2int,
        "int2word": int2word,
        "specials": special_tokens,
        "min_freq": min_freq
    }

    with open(VOCAB_OUTPUT, "wb") as f:
        pickle.dump(vocab, f)

    print(f"Flickr30k vocab size: {len(word2int)}")
    return vocab


def build_vocab_from_hf_dataset(train_ds, caption_field="caption_0", min_freq=2):
    token_counter = Counter()
    for example in train_ds:
        tokens = tokenize(example[caption_field])
        token_counter.update(tokens)
    
    tokens = [token for token, freq in token_counter.items() if freq >= min_freq]

    special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
    unique_tokens = special_tokens + sorted(tokens)

    word2int = {word: idx for idx, word in enumerate(unique_tokens)}
    int2word = {idx: word for word, idx in word2int.items()}

    vocab = {
        "word2int": word2int,
        "int2word": int2word,
        "specials": special_tokens,
        "min_freq": min_freq
    }

    with open(VOCAB_OUTPUT, "wb") as f:
        pickle.dump(vocab, f)

    print(f"Vocabulary size: {len(word2int)}")
    return vocab


def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def evaluate_recall_at_k_text2image(IMG_EMB_PATH, TXT_EMB_PATH, top_k=5):
    image_embs = normalize(np.load(IMG_EMB_PATH))
    text_embs = normalize(np.load(TXT_EMB_PATH))

    index = faiss.IndexFlatIP(image_embs.shape[1])
    index.add(image_embs)
    
    scores, indices = index.search(text_embs, top_k)
    correct_image_indices = np.repeat(np.arange(len(image_embs)), top_k)

    recall_at_k = (correct_image_indices[:, None] == indices).any(axis=1).mean()
    print(f"Recall@{top_k}: {recall_at_k:.4f}")


def evaluate_recall_at_k_generic(query_embs,db_embs,query_ids,db_ids,top_k=5):
    query_embs = normalize(query_embs)
    db_embs = normalize(db_embs)

    index = faiss.IndexFlatIP(db_embs.shape[1])
    index.add(db_embs)

    _, retrieved_idx = index.search(query_embs, top_k)

    hits = 0
    for q_idx, neighbors in enumerate(retrieved_idx):
        q_id = query_ids[q_idx]
        retrieved_ids = [db_ids[i] for i in neighbors]

        if q_id in retrieved_ids:
            hits += 1

    recall_at_k = hits / len(query_ids)
    print(f"Recall@{top_k}: {recall_at_k:.4f}")
    return recall_at_k


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


def faiss_index_build(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def show_unique_images_with_captions(indices, title, max_images=5, wrap_width=35):
    seen = set()
    unique_items = []

    for idx in indices:
        img_path = metadata[idx]["image_path"]
        caption  = metadata[idx]["caption"]

        if img_path not in seen:
            seen.add(img_path)
            unique_items.append((img_path, caption))

        if len(unique_items) == max_images:
            break

    n = len(unique_items)
    plt.figure(figsize=(4 * n, 5))

    for i, (img_path, caption) in enumerate(unique_items):
        plt.subplot(1, n, i + 1)
        img = Image.open(img_path).convert("RGB")
        plt.imshow(img)
        plt.axis("off")

        # Wrap caption so it fits nicely
        wrapped_caption = "\n".join(textwrap.wrap(caption, wrap_width))
        plt.title(wrapped_caption, fontsize=9)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def text_to_image(query_idx, k=5):
    query_text = metadata[query_idx]["caption"]
    print("Query text:", query_text)
    q_emb = text_embs[query_idx].reshape(1, -1)
    img_index = faiss_index_build(image_embs)
    _, indices = img_index.search(q_emb, k)
    show_unique_images_with_captions(indices[0], "Text → Image (Top-5, Unique with Captions). Query text: "+query_text)
    #show_images(indices[0], "Text → Image (Top-5)")


def image_to_text(query_idx, k=5):
    query_text = metadata[query_idx]["caption"]
    img = Image.open(metadata[query_idx]["image_path"]).convert("RGB")
    plt.imshow(img)
    plt.axis("off")
    plt.title("Query text:" + query_text)
    plt.show()
    q_emb = image_embs[query_idx].reshape(1, -1)
    txt_index = faiss_index_build(text_embs)
    _, indices = txt_index.search(q_emb, k)
    print("Top-5 retrieved captions:")
    for i in indices[0]:
        print("-", metadata[i]["caption"])


def text_to_text(query_idx, k=5):
    print("Query caption:")
    print(metadata[query_idx]["caption"])
    print("\nTop-5 similar captions:")

    q_emb = text_embs[query_idx].reshape(1, -1)
    txt_index = faiss_index_build(text_embs)
    _, indices = txt_index.search(q_emb, k + 1)  # +1 to include self

    for idx in indices[0][1:]:  # skip self
        print("-", metadata[idx]["caption"])


def image_to_image(query_idx, k=5):
    query_text = metadata[query_idx]["caption"]
    img = Image.open(metadata[query_idx]["image_path"]).convert("RGB")
    plt.imshow(img)
    plt.axis("off")
    plt.title("Query text:" + query_text)
    plt.show()

    q_emb = image_embs[query_idx].reshape(1, -1)
    img_index = faiss_index_build(image_embs)
    _, indices = img_index.search(q_emb, k)

    show_unique_images_with_captions(indices[0], "Image → Image (Top-5, Unique)")
    # show_images(indices[0], "Image → Image (Top-5)")


def generate_retrieval_embeddings_kaggle(image_encoder, text_encoder, df, vocab, image_root, device, batch_size=128, save_dir=None, split_name="test"):
    from dataloader_hf import Flickr30kDataset, collate_fn

    dataset = Flickr30kDataset(
        df=df,
        vocab=vocab,
        image_root=image_root
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    image_encoder.eval()
    text_encoder.eval()

    all_img_embs = []
    all_txt_embs = []
    all_txt_ids = []

    with torch.no_grad():
        for images, captions, img_ids, lengths in loader:
            images = images.to(device)
            captions = captions.to(device)

            img_emb = image_encoder(images)
            txt_emb = text_encoder(captions, lengths)

            all_img_embs.append(img_emb.cpu())
            all_txt_embs.append(txt_emb.cpu())
            all_txt_ids.extend(img_ids)

    image_embs = torch.cat(all_img_embs, dim=0).numpy()
    text_embs = torch.cat(all_txt_embs, dim=0).numpy()
    text_ids = np.array(all_txt_ids)

    # ---------------- SAVE ----------------
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, f"{split_name}_image_embeddings.npy"), image_embs)
        np.save(os.path.join(save_dir, f"{split_name}_text_embeddings.npy"), text_embs)
        np.save(os.path.join(save_dir, f"{split_name}_text_ids.npy"), text_ids)

        print(f"Saved embeddings to: {save_dir}")

    return image_embs, text_embs, text_ids


# Example usage
if __name__ == "__main__":
    # text_to_image(query_idx=0, k=50)
    # image_to_text(query_idx=0, k=5)
    # text_to_text(query_idx=0, k=5)
    # image_to_image(query_idx=0, k=50)
    
    # build_vocab(
    #     csv_path= CSV_FILE,
    #     output_path= VOCAB_OUTPUT
    # )

    ds = load_dataset("jxie/flickr8k")
    vocab = build_vocab_from_hf_dataset(ds["train"])

    # ds_30k = load_dataset("nlphuji/flickr30k-parquet")

    # vocab = build_vocab_from_hf_dataset(ds_30k["train"],min_freq=2)
    # print(vocab)

    # download_hf_dataset(
    #     repo_id=REPO_ID,  
    #     repo_type=REPO_TYPE,
    #     local_dir=LOCAL_DOWNLOAD_DIR
    # )

    # generate_flickr30k_clip_embeddings(
    #     image_dir=str(IMAGE_DIR),
    #     annotation_csv=str(ANNOTATION_CSV),
    #     output_dir=str(OUTPUT_DIR),
    #     batch_size=64
    # )

    # recall_i2i = evaluate_recall_at_k_generic(
    # query_embs=image_embs,
    # db_embs=image_embs,
    # query_ids=image_ids,
    # db_ids=image_ids,
    # top_k=1
    # )

    # recall_i2i = evaluate_recall_at_k_generic(
    # query_embs=image_embs,
    # db_embs=image_embs,
    # query_ids=image_ids,
    # db_ids=image_ids,
    # top_k=5
    # )

    # recall_i2i = evaluate_recall_at_k_generic(
    # query_embs=image_embs,
    # db_embs=image_embs,
    # query_ids=image_ids,
    # db_ids=image_ids,
    # top_k=10
    # )





