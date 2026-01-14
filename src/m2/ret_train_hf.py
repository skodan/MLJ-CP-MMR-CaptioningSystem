import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ret_model import ImageEncoder, TextEncoder
from dataloader_hf import FlickrDataset, collate_fn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "m2_resnet_lstm_attn.pth"


def contrastive_loss(im_emb, tex_emb, temperature=0.07):
    # Calculate similarity matrix (Batch x Batch)
    logits = (im_emb @ tex_emb.T) / temperature
    
    # Ground truth: the diagonal (image i matches text i)
    labels = torch.arange(im_emb.size(0)).to(im_emb.device)
    
    # Symmetric loss: image-to-text and text-to-image
    loss_i = nn.CrossEntropyLoss()(logits, labels)
    loss_t = nn.CrossEntropyLoss()(logits.T, labels)
    
    return (loss_i + loss_t) / 2


def init_m2_model_and_optimizer(vocab_size, device, lr=1e-4):
    # ---- Create models ----
    image_encoder = ImageEncoder().to(device)
    text_encoder = TextEncoder(vocab_size=vocab_size).to(device)

    # ---- Freeze entire ResNet backbone ----
    for param in image_encoder.backbone.parameters():
        param.requires_grad = False

    # ---- Unfreeze last 2–3 ResNet layers (layer2, layer3, layer4) ----
    for param in image_encoder.backbone[-3:].parameters():
        param.requires_grad = True

    # ---- Always train projection head ----
    for param in image_encoder.fc.parameters():
        param.requires_grad = True

    # ---- Optimizer: ONLY trainable parameters ----
    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            list(image_encoder.parameters()) +
            list(text_encoder.parameters())
        ),
        lr=lr
    )

    return image_encoder, text_encoder, optimizer



def init_m2_m3_m4_model_and_optimizer(vocab_size, device, lr=1e-4):
    image_encoder = ImageEncoder().to(device)
    text_encoder = TextEncoder(vocab_size=vocab_size).to(device)

    optimizer = torch.optim.Adam(
        list(image_encoder.parameters()) +
        list(text_encoder.parameters()),
        lr=lr
    )

    return image_encoder, text_encoder, optimizer


def run_epoch(loader, image_encoder, text_encoder, optimizer=None, device=device):
    total_loss = 0
    is_train = optimizer is not None

    for images, captions, lengths in loader:
        images, captions = images.to(device), captions.to(device)

        img_emb = image_encoder(images)
        txt_emb = text_encoder(captions, lengths)

        loss = contrastive_loss(img_emb, txt_emb)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
            img_embs.append(image_encoder(images).cpu())
            txt_embs.append(text_encoder(captions, lengths).cpu())

    return torch.cat(img_embs).numpy(), torch.cat(txt_embs).numpy()


# Commented out old training function
# def train_dual_encoder_hf(dataset, vocab_path, output_path, caption_field="caption_0", num_epochs=5, batch_size=32, lr=1e-4):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     dataset = FlickrDataset(dataset, vocab_path=vocab_path, caption_field=caption_field)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

#     image_encoder = ImageEncoder().to(device)
#     text_encoder = TextEncoder(vocab_size=len(dataset.word2int)).to(device)

#     params = list(text_encoder.parameters()) + list(image_encoder.fc.parameters())
#     optimizer = torch.optim.Adam(params, lr=lr)
#     #criterion = nn.MSELoss()

#     for epoch in range(num_epochs):
#         image_encoder.train()
#         text_encoder.train()
#         total_loss = 0
        
#         for images, captions, lengths in loader:
#             images, captions = images.to(device), captions.to(device)
#             image_emb = image_encoder(images)
#             text_emb = text_encoder(captions, lengths)

#             loss = contrastive_loss(image_emb, text_emb)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(loader):.4f}")


#     torch.save({
#         'image_encoder': image_encoder.state_dict(),
#         'text_encoder': text_encoder.state_dict()
#     }, output_path)
#     print("Model saved to", output_path)