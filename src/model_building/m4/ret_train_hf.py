import os
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from ret_model import ImageEncoder, TextEncoderWithAttention
#from dataloader_hf import FlickrDataset, collate_fn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "m4_vit_lstm_attn.pth"


def contrastive_loss(im_emb, tex_emb, temperature=0.2):
    # Calculate similarity matrix (Batch x Batch)
    im_emb = F.normalize(im_emb, p=2, dim=1)
    tex_emb = F.normalize(tex_emb, p=2, dim=1)

    logits = (im_emb @ tex_emb.T) / temperature
    
    # Ground truth: the diagonal (image i matches text i)
    labels = torch.arange(im_emb.size(0)).to(im_emb.device)
    
    # Symmetric loss: image-to-text and text-to-image
    loss_i = nn.CrossEntropyLoss()(logits, labels)
    loss_t = nn.CrossEntropyLoss()(logits.T, labels)
    
    return (loss_i + loss_t) / 2


# def init_m2_m3_m4_model_and_optimizer(vocab_size, device, lr=1e-4):
#     image_encoder = ImageEncoder().to(device)
#     text_encoder = TextEncoder(vocab_size=vocab_size).to(device)

#     optimizer = torch.optim.Adam(
#         list(image_encoder.parameters()) +
#         list(text_encoder.parameters()),
#         lr=lr
#     )

#     return image_encoder, text_encoder, optimizer


def init_m4_model_and_optimizer(vocab_size, device, lr=1e-4):
    image_encoder = ImageEncoder().to(device)
    text_encoder = TextEncoderWithAttention(vocab_size=vocab_size).to(device)

    # ---------------------------
    # Freeze ViT backbone
    # ---------------------------
    for p in image_encoder.backbone.parameters():
        p.requires_grad = False

    # Unfreeze last ViT block
    for p in image_encoder.backbone.encoder.layers[-1].parameters():
        p.requires_grad = True

    # Projection head stays trainable
    for p in image_encoder.fc.parameters():
        p.requires_grad = True

    # ---------------------------
    # Optimizer: ONLY trainable params
    # ---------------------------
    trainable_params = (
        list(filter(lambda p: p.requires_grad, image_encoder.parameters())) +
        list(text_encoder.parameters())
    )

    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    return image_encoder, text_encoder, optimizer



# def run_epoch(loader, image_encoder, text_encoder, optimizer=None, device=device):
#     total_loss = 0
#     is_train = optimizer is not None

#     if is_train:
#         image_encoder.train()
#         text_encoder.train()
#     else:
#         image_encoder.eval()
#         text_encoder.eval()

#     for images, captions, lengths in loader:
#         images, captions, lengths = images.to(device), captions.to(device), lengths.to(device)

#         img_emb, image_feats = image_encoder(images)
 
#         txt_emb = text_encoder(captions, lengths, image_feats)

#         loss = contrastive_loss(img_emb, txt_emb)

#         if is_train:
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         total_loss += loss.item()

#     return total_loss / len(loader)


def run_epoch(loader, image_encoder, text_encoder, optimizer=None, device=device):
    total_loss = 0.0
    is_train = optimizer is not None

    if is_train:
        image_encoder.train()
        text_encoder.train()
    else:
        image_encoder.eval()
        text_encoder.eval()

    for images, captions, lengths in loader:
        images, captions = images.to(device), captions.to(device)

        if is_train:
            optimizer.zero_grad()

            img_emb, image_feats = image_encoder(images)
            txt_emb = text_encoder(captions, lengths, image_feats)

            loss = contrastive_loss(img_emb, txt_emb)
            loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                img_emb, image_feats = image_encoder(images)
                txt_emb = text_encoder(captions, lengths, image_feats)
                loss = contrastive_loss(img_emb, txt_emb)

        total_loss += loss.item()

    return total_loss / len(loader)


def train_dual_encoder_hf(image_encoder,text_encoder,train_loader,val_loader,test_loader,optimizer,num_epochs,device,save_path=OUTPUT_DIR):
    train_losses, val_losses, test_losses = [], [], []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss = run_epoch(
            train_loader, image_encoder, text_encoder, optimizer, device
        )

        with torch.no_grad():
            val_loss = run_epoch(
                val_loader, image_encoder, text_encoder, None, device
            )
            test_loss = run_epoch(
                test_loader, image_encoder, text_encoder, None, device
            )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            
            torch.save(
                {
                    "image_encoder": image_encoder.state_dict(),
                    "text_encoder": text_encoder.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": best_val_loss
                },
                save_path
            )

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Test: {test_loss:.4f}"
        )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_losses": test_losses
    }


def extract_embeddings(loader, image_encoder, text_encoder):
    img_embs, txt_embs = [], []

    image_encoder.eval()
    text_encoder.eval()

    with torch.no_grad():
        for images, captions, lengths in loader:
            images, captions = images.to(device), captions.to(device)

            img_emb, image_feats = image_encoder(images)
            txt_emb = text_encoder(captions, lengths, image_feats)

            img_embs.append(img_emb.cpu())
            txt_embs.append(txt_emb.cpu())

    return torch.cat(img_embs).numpy(), torch.cat(txt_embs).numpy()