import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights


class ImageEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super(ImageEncoder, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1] # remove final fc layer
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)


    def forward(self, images):
        with torch.no_grad():
            features = self.backbone(images).squeeze()
        embeddings = self.fc(features)
        return embeddings


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size=300, hidden_size=512, output_dim=512):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)


    def forward(self, captions, lengths):
        embedded = self.embedding(captions)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        out = self.fc(hidden[-1])
        return out