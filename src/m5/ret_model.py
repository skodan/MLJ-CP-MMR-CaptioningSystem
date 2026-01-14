import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from transformers import CLIPTextModel, CLIPTokenizer

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512, proj_dim=512):
        super().__init__()

        vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        vit.heads = nn.Identity()
        self.backbone = vit

        self.proj = nn.Sequential(
        nn.Linear(vit.hidden_dim, embed_dim),
        nn.GELU(),
        nn.Linear(embed_dim, proj_dim),
        nn.LayerNorm(proj_dim)
        )


    def forward(self, images):
        x = self.backbone(images)       # (B, hidden_dim)
        x = self.proj(x)                # (B, proj_dim)
        return F.normalize(x, dim=1)


class CLIPTextEncoder(nn.Module):
    def __init__(self, proj_dim=512):
        super().__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.text_model = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.proj = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, proj_dim),
            nn.LayerNorm(proj_dim)
        )

    def forward(self, captions):
        tokens = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(next(self.parameters()).device)

        outputs = self.text_model(**tokens)
        cls_emb = outputs.last_hidden_state[:, 0]  # CLS token

        x = self.proj(cls_emb)
        return F.normalize(x, dim=1)