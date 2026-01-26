import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import faiss
import pickle
from typing import Tuple

# Import your updated models (adjust paths as needed)
from models.resnet_lstm_attention.ret_mod_defs import ImageEncoderViT, TextEncoder  # ViT image encoder + LSTM text encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-5  # low LR for ViT
TEMPERATURE = 0.2
EMBED_DIM = 512
SAVE_PATH = OUTPUT_DIR / "vit_retrieval_model.pth"
BEST_MODEL_PATH = OUTPUT_DIR / "vit_retrieval_best.pth"

# Load full Flickr8k dataset (train + val + test)
print("Loading full Flickr8k dataset...")
ds_dict = load_dataset("jxie/flickr8k")
dataset = concatenate_datasets([ds_dict["train"], ds_dict["validation"], ds_dict["test"]])
print(f"Loaded {len(dataset)} image-caption pairs.")

# Simple transform (ViT expects 224x224)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset
class FlickrRetrievalDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.transform(item["image"])
        caption = item["captions"][0]  # Use first caption
        return image, caption

train_dataset = FlickrRetrievalDataset(dataset, transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# Models
image_encoder = ImageEncoderViT(embed_dim=EMBED_DIM).to(device)
text_encoder = TextEncoder(vocab_size=10000, embed_dim=300, hidden_dim=512, out_dim=EMBED_DIM).to(device)  # adjust vocab_size

# Freeze most of ViT (fine-tune last 2 blocks)
for name, param in image_encoder.backbone.named_parameters():
    if "blocks.10" not in name and "blocks.11" not in name and "norm" not in name:
        param.requires_grad = False

print("Number of trainable parameters:")
print(f"Image Encoder: {sum(p.numel() for p in image_encoder.parameters() if p.requires_grad)}")
print(f"Text Encoder: {sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)}")

optimizer = torch.optim.AdamW(
    list(image_encoder.parameters()) + list(text_encoder.parameters()),
    lr=LR,
    weight_decay=0.05
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Contrastive loss
def contrastive_loss(img_emb, txt_emb, temperature=TEMPERATURE):
    img_emb = F.normalize(img_emb, p=2, dim=1)
    txt_emb = F.normalize(txt_emb, p=2, dim=1)

    logits = (img_emb @ txt_emb.T) / temperature
    labels = torch.arange(img_emb.size(0)).to(device)

    loss_i = nn.CrossEntropyLoss()(logits, labels)
    loss_t = nn.CrossEntropyLoss()(logits.T, labels)
    return (loss_i + loss_t) / 2

# Training loop
best_val_loss = float('inf')
train_losses = []

for epoch in range(EPOCHS):
    image_encoder.train()
    text_encoder.train()
    epoch_loss = 0.0

    for images, captions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = images.to(device)
        # Tokenize captions (you need to add a tokenizer here - reuse simple_tokenize from utils.py)
        # For simplicity, assume you have a function tokenize_batch(captions) → (tokens, lengths)
        # tokens, lengths = tokenize_batch(captions)
        # tokens, lengths = tokens.to(device), lengths.to(device)

        # Placeholder - replace with your actual tokenization
        # tokens = torch.randint(0, 10000, (images.size(0), 20)).to(device)
        # lengths = torch.ones(images.size(0), dtype=torch.long) * 20

        img_emb = image_encoder(images)
        txt_emb = text_encoder(tokens, lengths)  # update this line with your actual call

        loss = contrastive_loss(img_emb, txt_emb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save best model
    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        torch.save({
            "image_encoder": image_encoder.state_dict(),
            "text_encoder": text_encoder.state_dict(),
            "epoch": epoch,
            "loss": avg_loss
        }, BEST_MODEL_PATH)
        print(f"Best model saved at epoch {epoch+1} (Loss: {avg_loss:.4f})")

# Final save
torch.save({
    "image_encoder": image_encoder.state_dict(),
    "text_encoder": text_encoder.state_dict(),
    "train_losses": train_losses
}, SAVE_PATH)
print("Training completed. Final model saved.")

# Generate embeddings for FAISS (run after training)
def extract_embeddings(loader, image_encoder, text_encoder):
    img_embs, txt_embs = [], []

    image_encoder.eval()
    text_encoder.eval()

    with torch.no_grad():
        for images, captions in tqdm(loader, desc="Extracting embeddings"):
            images = images.to(device)
            # tokens, lengths = tokenize_batch(captions)
            # tokens, lengths = tokens.to(device), lengths.to(device)

            img_emb = image_encoder(images)
            txt_emb = text_encoder(tokens, lengths)  # update with your call

            img_embs.append(img_emb.cpu().numpy())
            txt_embs.append(txt_emb.cpu().numpy())

    return np.concatenate(img_embs), np.concatenate(txt_embs)

# Run extraction on full dataset
full_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
img_embs, txt_embs = extract_embeddings(full_loader, image_encoder, text_encoder)

# Save embeddings and indices
faiss.write_index(faiss.IndexFlatIP(EMBED_DIM), str(OUTPUT_DIR / "image_embeddings.faiss"))
faiss.write_index(faiss.IndexFlatIP(EMBED_DIM), str(OUTPUT_DIR / "text_embeddings.faiss"))

with open(OUTPUT_DIR / "image_id_map.pkl", "wb") as f:
    pickle.dump({i: i for i in range(len(dataset))}, f)  # adjust mapping as needed

with open(OUTPUT_DIR / "text_id_map.pkl", "wb") as f:
    pickle.dump({i: dataset[i]["captions"][0] for i in range(len(dataset))}, f)

print("Embeddings and indices saved successfully.")