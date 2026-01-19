from matplotlib import text
import torch
from .ret_mod_defs import ImageEncoder, TextEncoder
from .utils import simple_tokenize


print("LOADED clip_loader.py - UPDATED VERSION with encode_text(text: str)")

class CLIPRetrievalModel:
    """
    Wrapper to expose a CLIP-like interface:
    - encode_image(images)
    - encode_text(captions, lengths)
    """

    def __init__(self, image_encoder, text_encoder, vocab, max_caption_len=30):
        print("CLIPRetrievalModel initialized - single-arg encode_text version")
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.vocab = vocab
        self.max_caption_len = max_caption_len

    @torch.no_grad()
    def encode_image(self, images):
        return self.image_encoder(images)

    @torch.no_grad()
    def encode_text(self, text: str):
        """
        Accept raw text string → tokenize → pad → encode
        No need to pass 'lengths' from outside anymore
        """
        # Tokenize using the same logic as training
        words = [self.vocab.get(t, self.vocab['<unk>']) for t in simple_tokenize(text.lower())]
        tokens = words[:self.max_caption_len]
        tokens = [self.vocab['<start>']] + tokens + [self.vocab['<end>']]
        length = len(tokens)
        padded = tokens + [self.vocab['<pad>']] * (self.max_caption_len + 2 - length)

        captions = torch.tensor([padded], dtype=torch.long).to(self.text_encoder.embedding.weight.device)
        lengths = torch.tensor([length], dtype=torch.long).to(self.text_encoder.embedding.weight.device)

        return self.text_encoder(captions, lengths)


def load_clip_model(model_path: str, vocab: dict, device: torch.device = torch.device("cpu")):
    checkpoint = torch.load(model_path, map_location=device)

    image_encoder = ImageEncoder(embed_dim=512).to(device)

    # IMPORTANT: use vocab_size from checkpoint
    text_encoder = TextEncoder(
        vocab_size=checkpoint["vocab_size"]
    ).to(device)

    image_encoder.load_state_dict(checkpoint["image_encoder_state_dict"])
    text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"])

    image_encoder.eval()
    text_encoder.eval()

    return CLIPRetrievalModel(image_encoder, text_encoder, vocab)

