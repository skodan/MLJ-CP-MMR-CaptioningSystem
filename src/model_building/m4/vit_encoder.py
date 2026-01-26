import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vit.children())[:-2])  # Remove head
        for param in self.features.parameters():
            param.requires_grad = False  # Freeze ViT for efficiency

    def forward(self, images):
        features = self.features(images)  # (B, seq_len, embed_dim)
        return features  # Already in shape for attention (B, 197, 768 for base)


class ViTImageEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(vit.children())[:-2])
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pool over sequence
        self.fc = nn.Linear(vit.embed_dim, embed_dim)  # 768 → 512

    def forward(self, images):
        feat = self.backbone(images)  # (B, seq_len, embed_dim)
        feat = feat.mean(dim=1)  # Global average pool
        emb = self.fc(feat)
        return torch.nn.functional.normalize(emb, p=2, dim=1)