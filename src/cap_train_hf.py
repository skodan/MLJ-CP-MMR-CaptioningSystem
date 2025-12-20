import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader_hf import FlickrDataset, collate_fn

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, captions, lengths in dataloader:
        images = images.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()

        outputs = model(images, captions, lengths)

        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            captions[:, 1:].reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, captions, lengths in dataloader:
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions, lengths)

            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                captions[:, 1:].reshape(-1)
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)


def fit_model_hf(model,train_ds,val_ds,test_ds,vocab_path,output_model_path,batch_size=32,num_epochs=5,lr=1e-4,device=None,save_best_only=True):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    train_dataset = FlickrDataset(train_ds, vocab_path=vocab_path)
    val_dataset   = FlickrDataset(val_ds, vocab_path=vocab_path)
    test_dataset  = FlickrDataset(test_ds, vocab_path=vocab_path)

    PAD_IDX = train_dataset.pad_idx

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

 
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": []
    }

    best_val_loss = float("inf")


    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = eval_one_epoch(
            model, val_loader, criterion, device
        )
        test_loss = eval_one_epoch(
            model, test_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Test: {test_loss:.4f}"
        )

        if save_best_only:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_model_path)
        else:
            torch.save(model.state_dict(), output_model_path)

    print(f"Model saved to: {output_model_path}")
    return history