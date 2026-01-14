import math
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ret_model import ImageEncoder, CLIPTextEncoder
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "m5_ViT_Transformer.pth"


def freeze_vit_except_last_n(image_encoder, n=2):
    for p in image_encoder.backbone.parameters():
        p.requires_grad = False

    for block in image_encoder.backbone.encoder.layers[-n:]:
        for p in block.parameters():
            p.requires_grad = True

    for p in image_encoder.proj.parameters():
        p.requires_grad = True


def freeze_clip_text_encoder(text_encoder):
    for p in text_encoder.text_model.parameters():
        p.requires_grad = False


def contrastive_loss(im_emb, txt_emb, logit_scale):
    scale = torch.clamp(logit_scale.exp(), max=100)
    logits = scale * (im_emb @ txt_emb.T)

    labels = torch.arange(im_emb.size(0), device=im_emb.device)

    loss_i = nn.CrossEntropyLoss()(logits, labels)
    loss_t = nn.CrossEntropyLoss()(logits.T, labels)

    return (loss_i + loss_t) / 2


def init_m5_model_and_optimizer(device):
    image_encoder = ImageEncoder().to(device)
    text_encoder = CLIPTextEncoder().to(device)

    logit_scale = nn.Parameter(
    torch.tensor(math.log(1 / 0.1), device=device)
    )

    optimizer = torch.optim.AdamW(
    [
        {
        "params": text_encoder.text_model.text_model.encoder.layers[-2:].parameters(),
        "lr": 1e-5
        },
        {
            "params": image_encoder.proj.parameters(),
            "lr": 3e-4
        },
        {
            "params": text_encoder.proj.parameters(),
            "lr": 3e-4
        },
        {
            "params": [logit_scale],
            "lr": 1e-4
        }
    ],
    weight_decay=1e-4
    )

    return image_encoder, text_encoder, logit_scale, optimizer


def run_epoch(loader, image_encoder, text_encoder, logit_scale, optimizer=None, device=device):
    total_loss = 0.0
    is_train = optimizer is not None

    if is_train:
        image_encoder.train()
        text_encoder.train()
    else:
        image_encoder.eval()
        text_encoder.eval()

    for images, captions, _ in loader:
        images = images.to(device)

        # captions must be list[str]
        img_emb = image_encoder(images)
        txt_emb = text_encoder(captions)

        loss = contrastive_loss(img_emb, txt_emb, logit_scale)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train_dual_encoder_hf(image_encoder, text_encoder, logit_scale, train_loader, val_loader, test_loader, optimizer, num_epochs, device, save_path):
    train_losses, val_losses, test_losses = [], [], []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss = run_epoch(train_loader, image_encoder, text_encoder, logit_scale, optimizer, device)

        with torch.no_grad():
            val_loss = run_epoch(val_loader, image_encoder, text_encoder, logit_scale, None, device)
            test_loss = run_epoch(test_loader, image_encoder, text_encoder, logit_scale, None, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "image_encoder": image_encoder.state_dict(),
                    "text_encoder": text_encoder.state_dict(),
                    "logit_scale": logit_scale.detach().cpu(),
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


def extract_embeddings(loader, image_encoder, text_encoder, device):
    img_embs, txt_embs = [], []

    image_encoder.eval()
    text_encoder.eval()

    with torch.no_grad():
        for images, captions, _ in loader:
            images = images.to(device)

            img_embs.append(image_encoder(images).cpu())
            txt_embs.append(text_encoder(captions).cpu())

    return torch.cat(img_embs).numpy(), torch.cat(txt_embs).numpy()