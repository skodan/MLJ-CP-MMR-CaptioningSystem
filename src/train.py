import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import pickle
from model import ImageEncoder, TextEncoder
from dataloader import FlickrDataset, collate_fn

def train_dual_encoder(train_csv, image_dir, vocab_path, output_path, num_epochs=5, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(train_csv)
    with open(vocab_path, "rb") as f:
        word2int, _ = pickle.load(f)
    
    dataset = FlickrDataset(df, word2int, image_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    image_encoder = ImageEncoder().to(device)
    text_encoder = TextEncoder(vocab_size=len(word2int)).to(device)

    params = list(text_encoder.parameters()) + list(image_encoder.fc.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        image_encoder.train()
        text_encoder.train()
        total_loss = 0
        
        for images, captions, lengths in loader:
            images, captions = images.to(device), captions.to(device)
            image_emb = image_encoder(images)
            text_emb = text_encoder(captions, lengths)

            loss = criterion(image_emb, text_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(loader):.4f}")


    torch.save({
        'image_encoder': image_encoder.state_dict(),
        'text_encoder': text_encoder.state_dict()
    }, output_path)
    print("Model saved to", output_path)