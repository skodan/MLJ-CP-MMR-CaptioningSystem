import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ImageEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()

        vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.backbone = vit
        self.backbone.heads = nn.Identity()  # remove classifier

        self.fc = nn.Linear(vit.hidden_dim, output_dim)

    def forward(self, images):
        features = self.backbone(images)      # (B, hidden_dim)
        embeddings = self.fc(features)        # (B, 512)
        return F.normalize(embeddings, p=2, dim=1)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size=512, hidden_size=512, num_layers=2, num_heads=8, output_dim=512, max_len=100):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_len, embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(embed_size, output_dim)

    def forward(self, captions, lengths):
        B, T = captions.size()
        device = captions.device

        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        x = self.embedding(captions) + self.pos_embedding(positions)

        # Padding mask: True for PAD tokens
        max_len = captions.size(1)
        mask = torch.arange(max_len, device=device).expand(B, max_len)
        mask = mask >= torch.tensor(lengths, device=device).unsqueeze(1)

        out = self.transformer(
            x,
            src_key_padding_mask=mask
        )  # (B, T, E)

        # Mean pooling over valid tokens
        out = out.masked_fill(mask.unsqueeze(-1), 0.0)
        pooled = out.sum(dim=1) / torch.tensor(lengths, device=device).unsqueeze(1)

        embeddings = self.fc(pooled)
        return F.normalize(embeddings, p=2, dim=1)