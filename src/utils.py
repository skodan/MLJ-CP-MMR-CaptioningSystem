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

try:
    import faiss
except ImportError:
    faiss = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = PROJECT_ROOT / "data" / "captions" / "flickr8k_train.csv"
VOCAB_OUTPUT = PROJECT_ROOT / "data" / "captions" / "vocab.pkl"
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
    ds = load_dataset("jxie/flickr8k")
    vocab = build_vocab_from_hf_dataset(ds["train"])
