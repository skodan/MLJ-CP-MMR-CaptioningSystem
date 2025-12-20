import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetLSTMCaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.img_fc = nn.Linear(resnet.fc.in_features, embed_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions, lengths):
        with torch.no_grad():
            features = self.encoder(images)
        features = features.flatten(1)
        img_emb = self.img_fc(features)              # (B, embed_dim)

        word_embeddings = self.embedding(captions[:, :-1])  # (B, T-1, embed_dim)

        # prepend image embedding
        img_emb = img_emb.unsqueeze(1)               # (B, 1, embed_dim)
        lstm_input = torch.cat([img_emb, word_embeddings], dim=1)

        lstm_out, _ = self.lstm(lstm_input)
        outputs = self.fc(lstm_out)

        return outputs
