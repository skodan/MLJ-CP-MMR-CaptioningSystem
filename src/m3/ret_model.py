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
    def __init__(self, vocab_size, embed_size=300, hidden_size=512, output_dim=512):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_dim)


    def forward(self, captions, lengths):
        embedded = self.embedding(captions)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_out,_ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        attn_scores = self.attn(outputs).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(outputs * attn_weights.unsqueeze(-1), dim=1)
        embeddings = self.fc(context)
        return F.normalize(embeddings, p=2, dim=1)