import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import vit_b_16, ViT_B_16_Weights


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# class ImageEncoderViT(nn.Module):
#     def __init__(self, embed_dim=512):
#         super().__init__()
#         vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
#         self.backbone = nn.Sequential(*list(vit.children())[:-2])
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(vit.embed_dim, embed_dim)  # 768 → 512

#         # Freeze early layers
#         for child in list(self.backbone.children())[:7]:
#             for param in child.parameters():
#                 param.requires_grad = False

#     def forward(self, images):
#         feat = self.backbone(images)
#         feat = feat.mean(dim=1)
#         emb = self.fc(feat)
#         return F.normalize(emb, p=2, dim=1)
class ImageEncoderViT(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        # Load the model
        vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        
        # We take everything except the final classification heads
        self.backbone = nn.Sequential(*list(vit.children())[:-2])
        
        # FIX: Use 768 (the hidden_dim of vit_b_16)
        self.fc = nn.Linear(768, embed_dim) 

        # Optional: Freeze early layers for faster training
        for child in list(self.backbone.children())[:7]:
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, images):
        # Result from backbone: (Batch, 197, 768)
        feat = self.backbone(images)
        
        # Mean pool across the sequence/patch dimension
        feat = feat.mean(dim=1)
        
        # Project to your desired embedding space (512)
        emb = self.fc(feat)
        return F.normalize(emb, p=2, dim=1)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=512, out_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, captions, lengths):
        embedded = self.embedding(captions)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        attn_w = F.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_w * lstm_out, dim=1)

        emb = self.fc(context)
        return F.normalize(emb, p=2, dim=1)