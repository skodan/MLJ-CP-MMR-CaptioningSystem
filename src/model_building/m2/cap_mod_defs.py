import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, images):
        features = self.features(images)           # (B, 2048, 7, 7)
        features = features.permute(0,2,3,1)       # (B,7,7,2048)
        features = features.view(features.size(0), -1, 2048)  # (B,49,2048)
        return features


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)    # for decoder hidden
        self.W_e = nn.Linear(2048, hidden_dim, bias=False)          # for encoder features
        self.v   = nn.Linear(hidden_dim, 1, bias=False)             # final score

    def forward(self, hidden, encoder_outputs):
        # Project and add
        hidden_proj = self.W_h(hidden).unsqueeze(1)               # (B, 1, hidden_dim)
        encoder_proj = self.W_e(encoder_outputs)                  # (B, seq_len, hidden_dim)
        
        # Compute energy: tanh(W_h * h + W_e * e)
        energy = torch.tanh(hidden_proj + encoder_proj)           # (B, seq_len, hidden_dim)
        
        # Score: v^T * energy
        scores = self.v(energy).squeeze(2)                        # (B, seq_len)
        
        # Attention weights
        attn_weights = torch.softmax(scores, dim=1)               # (B, seq_len)
        
        # Context vector: weighted sum of encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)                              # (B, 2048)

        return context


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTMCell(embed_size + 2048, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, captions, encoder_outputs):
        embedded = self.embedding(captions[:, :-1])  # (B, seq_len-1, embed_size)
        
        h = torch.zeros(embedded.size(0), 512, device=device)
        c = torch.zeros(embedded.size(0), 512, device=device)
        
        outputs = []
        
        for t in range(embedded.size(1)):
            context = self.attention(h, encoder_outputs)          # ← now correct
            inp = torch.cat([embedded[:, t, :], context], dim=1)
            h, c = self.lstm(inp, (h, c))
            outputs.append(self.fc(h))
        
        return torch.stack(outputs, dim=1)  # (B, seq_len-1, vocab_size)

    @torch.no_grad()
    def generate(self, encoder_outputs, vocab, inv_vocab,max_len=25):
        h = torch.zeros(1, 512, device=device)
        c = torch.zeros(1, 512, device=device)
        
        token = torch.tensor([[vocab['<start>']]], device=device)
        caption = []
        
        for _ in range(max_len):
            emb = self.embedding(token)
            context = self.attention(h, encoder_outputs)
            inp = torch.cat([emb.squeeze(0), context], dim=1)
            h, c = self.lstm(inp, (h, c))
            pred = self.fc(h).argmax(1)
            caption.append(pred.item())
            token = pred.unsqueeze(0)
            
            if pred.item() == vocab['<end>']:
                break
        
        return [inv_vocab.get(i, '<unk>') for i in caption if i != vocab['<end>']]