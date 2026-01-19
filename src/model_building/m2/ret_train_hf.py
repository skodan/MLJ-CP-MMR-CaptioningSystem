import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ret_model import ImageEncoder, TextEncoder
from dataloader_hf import FlickrDataset, collate_fn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def contrastive_loss(img_emb, txt_emb, temp=0.07):
    sim = (img_emb @ txt_emb.T) / temp
    labels = torch.arange(len(img_emb), device=img_emb.device)
    loss_i2t = F.cross_entropy(sim, labels)
    loss_t2i = F.cross_entropy(sim.T, labels)
    return (loss_i2t + loss_t2i) / 2


def init_m2_model_and_optimizer(vocab_size, device):
    # ---- Create models ----
    image_encoder = ImageEncoder().to(device)
    text_encoder = TextEncoder(vocab_size=vocab_size).to(device)

    optimizer = optim.AdamW([
        {'params': image_encoder.parameters(), 'lr': 3e-5},
        {'params': text_encoder.parameters(),  'lr': 1e-4},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    return image_encoder, text_encoder, optimizer, scheduler


def run_epoch(loader, image_encoder, text_encoder, optimizer=None, device=device) -> float:
    total_loss = 0
    num_batches = len(loader)
    is_train = optimizer is not None

    if is_train:
        image_encoder.train()
        text_encoder.train()
    else:
        image_encoder.eval()
        text_encoder.eval()

    #progress = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False) if desc else loader

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            # Different datasets might return different number of items
            if len(batch) == 5:
                images, captions, lengths, _, _ = batch  # train/val with captions_str
            else:
                images, captions, lengths = batch

            images = images.to(device)
            captions = captions.to(device)
            lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)

            img_emb = image_encoder(images)
            txt_emb = text_encoder(captions, lengths)
            
            loss = contrastive_loss(img_emb, txt_emb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / num_batches


def print_epoch_summary(optimizer, epoch: int, num_epochs: int, train_loss: float, val_loss: float, test_loss: float, prev_lr: float = None):
    lr_info = ""
    if prev_lr is not None:
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < prev_lr:
            lr_info = f" (LR ↓ {prev_lr:.2e} → {current_lr:.2e})"
    
    print(f"Epoch [{epoch+1:2d}/{num_epochs:2d}] | "
          f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | Test: {test_loss:.4f}{lr_info}")


def print_final_results(train_losses, val_losses, test_losses):
    """Print nice table of all epochs results"""
    print("\n" + "="*60)
    print("Final Training Results")
    print("═"*60)
    print(f"{'Epoch':^6} | {'Train Loss':^12} | {'Val Loss':^12} | {'Test Loss':^12}")
    print("-"*60)
    
    for e in range(len(train_losses)):
        print(f"{e+1:6d} | {train_losses[e]:12.4f} | {val_losses[e]:12.4f} | {test_losses[e]:12.4f}")
    print("="*60)


def train_dual_encoder(train_loader,val_loader,test_loader,num_epochs,device,vocab_size,save_path) -> dict:
    """
    Main training function - clean and modular version
    Returns dictionary with losses history
    """
    # Initialize models, optimizer, scheduler
    image_encoder, text_encoder, optimizer, scheduler = init_m2_model_and_optimizer(vocab_size, device)

    train_losses = []
    val_losses = []
    test_losses = []
    
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # ─── Training ────────────────────────────────────────
        prev_lr = optimizer.param_groups[0]['lr']
        
        train_loss = run_epoch(train_loader, image_encoder, text_encoder, optimizer, device)

        # ─── Validation + Test ───────────────────────────────
        val_loss = run_epoch(val_loader, image_encoder, text_encoder, None, device)
        test_loss = run_epoch(test_loader, image_encoder, text_encoder, None, device)

        # Collect losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        # Scheduler step
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'image_encoder': image_encoder.state_dict(),
                'text_encoder': text_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, save_path)
            print(f"  → Saved best model (val_loss: {best_val_loss:.4f})")

        # Print progress
        print_epoch_summary(optimizer, epoch, num_epochs, train_loss, val_loss, test_loss, prev_lr)

    # Final summary table
    print_final_results(train_losses, val_losses, test_losses)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_losses": test_losses,
        "best_val_loss": best_val_loss,
        "image_encoder": image_encoder,
        "text_encoder": text_encoder
    }



# def train_dual_encoder_hf(image_encoder,text_encoder,train_loader,val_loader,test_loader,optimizer,num_epochs,device,save_path=OUTPUT_DIR):
#     train_losses, val_losses, test_losses = [], [], []
#     best_val_loss = float("inf")

#     for epoch in range(num_epochs):
#         train_loss = run_epoch(
#             train_loader, image_encoder, text_encoder, optimizer, device
#         )

#         with torch.no_grad():
#             val_loss = run_epoch(
#                 val_loader, image_encoder, text_encoder, None, device
#             )
#             test_loss = run_epoch(
#                 test_loader, image_encoder, text_encoder, None, device
#             )

#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
#         test_losses.append(test_loss)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss

#             save_dir = os.path.dirname(save_path)
#             if save_dir:
#                 os.makedirs(save_dir, exist_ok=True)
            
#             torch.save(
#                 {
#                     "image_encoder": image_encoder.state_dict(),
#                     "text_encoder": text_encoder.state_dict(),
#                     "epoch": epoch + 1,
#                     "val_loss": best_val_loss
#                 },
#                 save_path
#             )

#         print(
#             f"Epoch [{epoch+1}/{num_epochs}] | "
#             f"Train: {train_loss:.4f} | "
#             f"Val: {val_loss:.4f} | "
#             f"Test: {test_loss:.4f}"
#         )

#     return {
#         "train_losses": train_losses,
#         "val_losses": val_losses,
#         "test_losses": test_losses
#     }


# def extract_embeddings(loader, image_encoder, text_encoder):
#     img_embs, txt_embs = [], []

#     image_encoder.eval()
#     text_encoder.eval()

#     with torch.no_grad():
#         for images, captions, lengths in loader:
#             images, captions = images.to(device), captions.to(device)
#             img_embs.append(image_encoder(images).cpu())
#             txt_embs.append(text_encoder(captions, lengths).cpu())

#     return torch.cat(img_embs).numpy(), torch.cat(txt_embs).numpy()


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