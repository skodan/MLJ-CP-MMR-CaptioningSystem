import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super(ImageEncoder, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)

    # Forward method with gradient flow for fine-tuning
    def forward(self, images):
        features = self.backbone(images)
        features = features.flatten(1)
        embeddings = self.fc(features)
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