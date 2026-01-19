import torch
import pickle
from torchvision import transforms
from PIL import Image
from .cap_mod_defs import EncoderCNN, DecoderRNN  # reuse your exact classes

class CaptioningService:
    def __init__(self, model_path, vocab_path, config):
        self.device = torch.device("cpu")

        # Load vocab
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        # Load checkpoint
        ckpt = torch.load(model_path, map_location=self.device)

        self.encoder = EncoderCNN().to(self.device)
        self.encoder.load_state_dict(ckpt["encoder_state_dict"])
        self.encoder.eval()

        self.decoder = DecoderRNN(vocab_size=ckpt["vocab_size"]).to(self.device)
        self.decoder.load_state_dict(ckpt["decoder_state_dict"])
        self.decoder.eval()

        self.max_len = config["max_length"]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def generate_caption(self, image: Image.Image) -> str:
        image = self.transform(image).unsqueeze(0).to(self.device)
        features = self.encoder(image)
        tokens = self.decoder.generate(
            features,
            vocab=self.vocab,
            inv_vocab=self.inv_vocab,
            max_len=self.max_len
        )
        return " ".join(tokens)
