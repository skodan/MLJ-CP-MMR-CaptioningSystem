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

    def forward(self, x):
        x = self.backbone._process_input(x)
        n = x.shape[0]

        cls_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = self.backbone.encoder(x)          # (B, 1+P, D)

        cls_emb = x[:, 0, :]                  # (B, D)
        patch_embs = x[:, 1:, :]              # (B, P, D)

        cls_emb = self.fc(cls_emb)            # (B, 512)
        patch_embs = self.fc(patch_embs)      # (B, P, 512)

        return F.normalize(cls_emb, p=2, dim=1), patch_embs



class Attention(nn.Module):
    def __init__(self, image_dim, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(image_dim + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, image_feats, hidden):
        # hidden: (B, H) -> expand to (B, P, H) to match image patches
        hidden_expanded = hidden.unsqueeze(1).expand(-1, image_feats.size(1), -1)
        energy = torch.tanh(self.attn(torch.cat((image_feats, hidden_expanded), dim=2)))
        scores = self.v(energy).squeeze(2)
        alpha = torch.softmax(scores, dim=1)
        context = (image_feats * alpha.unsqueeze(2)).sum(dim=1)
        return context



class TextEncoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=512, image_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.attention = Attention(image_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim + image_dim, 512)

    def forward(self, captions, lengths, image_feats=None):
        embeddings = self.embedding(captions)

        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        text_feat = h_n.squeeze(0)  # (B, 512)

        if image_feats is not None:
            context = self.attention(image_feats, text_feat)
        else:
            context = torch.zeros_like(text_feat)
            # text-only path for retrieval
            # fused = torch.cat(
            #     [text_feat, torch.zeros_like(text_feat)], dim=1
            # )

        # attention: text queries image patches
        # context = self.attention(image_feats, text_feat)  # (B, 512)

        # fused = torch.cat([text_feat, context], dim=1)
        fused = torch.cat([text_feat, context], dim=1)
        out = self.fc(fused)

        return F.normalize(out, dim=1)





# class TextEncoderWithAttention(nn.Module):
#     def __init__(self, vocab_size, embed_dim=300, hidden_dim=512, image_dim=512):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.attention = Attention(image_dim, hidden_dim)

#         self.lstm = nn.LSTM(
#             embed_dim + image_dim,
#             hidden_dim,
#             batch_first=True
#         )

#         self.fc = nn.Linear(hidden_dim, hidden_dim)

#     def forward(self, captions, lengths, image_feats):
#         # ---- Encode text WITHOUT attention ----
#         embeddings = self.embedding(captions)
#         packed = nn.utils.rnn.pack_padded_sequence(
#             embeddings, lengths, batch_first=True, enforce_sorted=False
#         )
#         _, (h_n, _) = self.lstm(packed)
#         text_feat = h_n.squeeze(0)        # (B, H)

#         # ---- SINGLE attention pooling over image patches ----
#         B, P, D = image_feats.size()
#         text_rep = text_feat.unsqueeze(1).repeat(1, P, 1)

#         energy = torch.tanh(self.attn(torch.cat([image_feats, text_rep], dim=2)))
#         scores = self.v(energy).squeeze(2)
#         alpha = torch.softmax(scores, dim=1)

#         context = (image_feats * alpha.unsqueeze(2)).sum(dim=1)

#         fused = torch.cat([text_feat, context], dim=1)
#         out = self.fc(fused)

#         return F.normalize(out, dim=1)



# class Attention(nn.Module):
#     def __init__(self, image_dim, hidden_dim):
#         super().__init__()
#         self.attn = nn.Linear(image_dim + hidden_dim, hidden_dim)
#         self.v = nn.Linear(hidden_dim, 1)

#     def forward(self, image_feats, hidden):
#         """
#         image_feats: (B, P, D)
#         hidden: (B, H)
#         """
#         hidden = hidden.unsqueeze(1).repeat(1, image_feats.size(1), 1)
#         energy = torch.tanh(self.attn(torch.cat((image_feats, hidden), dim=2)))
#         scores = self.v(energy).squeeze(2)
#         alpha = torch.softmax(scores, dim=1)

#         context = (image_feats * alpha.unsqueeze(2)).sum(dim=1)
#         return context
