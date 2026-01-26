import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import vit_b_16, ViT_B_16_Weights


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class EncoderViT(nn.Module):
    def __init__(self):
        super().__init__()
        vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vit.children())[:-2])  # (B, 197, 768)
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, images):
        features = self.features(images)
        features = features.view(features.size(0), -1, 768)  # (B, 197, 768) for attention
        return features


class Attention(nn.Module):
    def __init__(self, encoder_dim=512, hidden_dim=512):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_e = nn.Linear(encoder_dim, hidden_dim, bias=False)
        self.v   = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (B, hidden_dim) -> (B, 1, hidden_dim)
        # encoder_outputs: (B, 197, 512)
        
        hidden_proj = self.W_h(hidden).unsqueeze(1)
        encoder_proj = self.W_e(encoder_outputs) # (B, 197, hidden_dim)
        
        energy = torch.tanh(hidden_proj + encoder_proj)
        scores = self.v(energy).squeeze(2) # (B, 197)
        attn_weights = torch.softmax(scores, dim=1) # (B, 197)
        
        # BMM requires: (B, 1, 197) @ (B, 197, 512) -> (B, 1, 512)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        return context.squeeze(1) # (B, 512)
    
# class Attention(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)    # for decoder hidden
#         self.W_e = nn.Linear(2048, hidden_dim, bias=False)          # for encoder features
#         self.v   = nn.Linear(hidden_dim, 1, bias=False)             # final score

#     def forward(self, hidden, encoder_outputs):
#         # Project and add
#         hidden_proj = self.W_h(hidden).unsqueeze(1)               # (B, 1, hidden_dim)
#         encoder_proj = self.W_e(encoder_outputs)                  # (B, seq_len, hidden_dim)
        
#         # Compute energy: tanh(W_h * h + W_e * e)
#         energy = torch.tanh(hidden_proj + encoder_proj)           # (B, seq_len, hidden_dim)
        
#         # Score: v^T * energy
#         scores = self.v(energy).squeeze(2)                        # (B, seq_len)
        
#         # Attention weights
#         attn_weights = torch.softmax(scores, dim=1)               # (B, seq_len)
        
#         # Context vector: weighted sum of encoder outputs
#         context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
#         context = context.squeeze(1)                              # (B, 2048)

#         return context


# class DecoderRNN(nn.Module):
#     def __init__(self, vocab_size, embed_size=256, hidden_size=512):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.attention = Attention(hidden_size)
#         self.lstm = nn.LSTMCell(embed_size + 2048, hidden_size)
#         self.fc = nn.Linear(hidden_size, vocab_size)

# class DecoderRNN(nn.Module):
#     def __init__(self, vocab_size, embed_dim=512, hidden_dim=512):  # ← change to 512
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)  # input_size=embed_dim=512
#         self.attn = nn.Linear(hidden_dim + embed_dim, hidden_dim)  # if attention uses embed_dim
#         self.fc = nn.Linear(hidden_dim, vocab_size)

#     def forward(self, captions, encoder_outputs):
#         embedded = self.embedding(captions[:, :-1])  # (B, seq_len-1, embed_size)
        
#         h = torch.zeros(embedded.size(0), 512, device=device)
#         c = torch.zeros(embedded.size(0), 512, device=device)
        
#         outputs = []
        
#         for t in range(embedded.size(1)):
#             context = self.attention(h, encoder_outputs)          # ← now correct
#             inp = torch.cat([embedded[:, t, :], context], dim=1)
#             h, c = self.lstm(inp, (h, c))
#             outputs.append(self.fc(h))
        
#         return torch.stack(outputs, dim=1)  # (B, seq_len-1, vocab_size)

# class DecoderRNN(nn.Module):
#     def __init__(self, vocab_size, encoder_dim=512, embed_dim=512, hidden_dim=512):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
#         # Pass the encoder_dim (512) here!
#         self.attention = Attention(encoder_dim, hidden_dim) 
        
#         # Input to LSTM is: word_embedding + attention_context
#         self.lstm = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim)
#         self.fc = nn.Linear(hidden_dim, vocab_size)

#     def forward(self, captions, encoder_features):
#         embedded = self.embedding(captions)  # (B, seq, 512)
        
#         # If encoder_features is pooled (B, 512)
#         context = self.attention(encoder_features, embedded)  # adjust attention call
        
#         # Or if you concatenate per timestep
#         lstm_input = torch.cat([embedded, context.unsqueeze(1).repeat(1, embedded.size(1), 1)], dim=2)
        
#         lstm_out, _ = self.lstm(lstm_input)
#         outputs = self.fc(lstm_out)
#         return outputs

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, encoder_dim=512, embed_dim=512, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = Attention(encoder_dim, hidden_dim)  # FIXED: Using 512
        
        # LSTM input: word embedding + context vector
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, captions, encoder_features):
        # captions: (B, seq_len-1)
        # encoder_features: (B, seq_len_vit, 512)
        
        batch_size = encoder_features.size(0)
        embedded = self.embedding(captions) # (B, seq_len-1, embed_dim)
        
        # Initialize LSTM states
        h = torch.zeros(batch_size, self.hidden_dim, device=encoder_features.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=encoder_features.device)
        
        outputs = []
        for t in range(embedded.size(1)):
            # Get attention context
            context = self.attention(h, encoder_features)
            
            # Combine current word and context
            lstm_input = torch.cat([embedded[:, t, :], context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            
            # Get logits
            output = self.fc(h)
            outputs.append(output)
            
        return torch.stack(outputs, dim=1) # (B, seq_len-1, vocab_size)
    

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