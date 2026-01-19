import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# class ImageEncoder(nn.Module):
#     def __init__(self, embed_dim=512):
#         super().__init__()
#         resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
#         self.backbone = nn.Sequential(*list(resnet.children())[:-2])
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(resnet.fc.in_features, embed_dim)

#         # Freeze early layers (up to layer2)
#         freeze_until = 6
#         for child in list(self.backbone.children())[:freeze_until]:
#             for p in child.parameters():
#                 p.requires_grad = False

#         print("ImageEncoder: early layers frozen.")

#     def forward(self, x):
#         feat = self.backbone(x)
#         feat = self.pool(feat)
#         feat = torch.flatten(feat, 1)
#         emb = self.fc(feat)
#         return F.normalize(emb, p=2, dim=1)


# class TextEncoder(nn.Module):
#     def __init__(self, vocab_size, embed_dim=300, hidden_dim=512, out_dim=512):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
#         self.attn = nn.Linear(hidden_dim, 1)
#         self.fc = nn.Linear(hidden_dim, out_dim)

#     def forward(self, captions, lengths):
#         embedded = self.embedding(captions)
#         packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(),
#                                                    batch_first=True, enforce_sorted=False)
#         lstm_out, _ = self.lstm(packed)
#         lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

#         attn_w = F.softmax(self.attn(lstm_out), dim=1)
#         context = torch.sum(attn_w * lstm_out, dim=1)

#         emb = self.fc(context)
#         return F.normalize(emb, p=2, dim=1)

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(resnet.fc.in_features, embed_dim)

        # Freeze early layers (up to layer2)
        freeze_until = 7 # Unfreezed more layers to improve the retrieval values
        for child in list(self.backbone.children())[:freeze_until]:
            for p in child.parameters():
                p.requires_grad = False

        print("ImageEncoder: layers 0-6 frozen, layer3+layer4 trainable.")

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)
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