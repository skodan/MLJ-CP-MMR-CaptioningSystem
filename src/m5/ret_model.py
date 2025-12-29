import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights


# --------------------------------------------------
# Image Encoder (ViT + Projection Head)
# --------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512, proj_dim=512):
        super().__init__()

        vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        vit.heads = nn.Identity()
        self.backbone = vit

        self.proj = nn.Sequential(
            nn.Linear(vit.hidden_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim)
        )

    def forward(self, images):
        x = self.backbone(images)       # (B, hidden_dim)
        x = self.proj(x)                # (B, proj_dim)
        return F.normalize(x, dim=1)


# --------------------------------------------------
# Text Encoder (Transformer + Projection Head)
# --------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size=512,
        num_layers=4,
        num_heads=8,
        ff_dim=2048,
        proj_dim=512,
        max_len=100
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_len, embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.proj = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, proj_dim)
        )

    def forward(self, captions, lengths):
        B, T = captions.size()
        device = captions.device

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = self.embedding(captions) + self.pos_embedding(pos)

        # Padding mask
        mask = torch.arange(T, device=device).expand(B, T) >= \
               torch.tensor(lengths, device=device).unsqueeze(1)

        x = self.transformer(x, src_key_padding_mask=mask)

        # Mean pooling over valid tokens
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        pooled = x.sum(dim=1) / torch.tensor(lengths, device=device).unsqueeze(1)

        x = self.proj(pooled)
        return F.normalize(x, dim=1)
