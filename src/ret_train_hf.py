import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ret_model import ImageEncoder, TextEncoder
from dataloader_hf import FlickrDataset, collate_fn

def contrastive_loss(im_emb, tex_emb, temperature=0.07):
    # Calculate similarity matrix (Batch x Batch)
    logits = (im_emb @ tex_emb.T) / temperature
    
    # Ground truth: the diagonal (image i matches text i)
    labels = torch.arange(im_emb.size(0)).to(im_emb.device)
    
    # Symmetric loss: image-to-text and text-to-image
    loss_i = nn.CrossEntropyLoss()(logits, labels)
    loss_t = nn.CrossEntropyLoss()(logits.T, labels)
    
    return (loss_i + loss_t) / 2


def train_dual_encoder_hf(dataset, vocab_path, output_path, caption_field="caption_0", num_epochs=5, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = FlickrDataset(dataset, vocab_path=vocab_path, caption_field=caption_field)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    image_encoder = ImageEncoder().to(device)
    text_encoder = TextEncoder(vocab_size=len(dataset.word2int)).to(device)

    params = list(text_encoder.parameters()) + list(image_encoder.fc.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    #criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        image_encoder.train()
        text_encoder.train()
        total_loss = 0
        
        for images, captions, lengths in loader:
            images, captions = images.to(device), captions.to(device)
            image_emb = image_encoder(images)
            text_emb = text_encoder(captions, lengths)

            loss = contrastive_loss(image_emb, text_emb)
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