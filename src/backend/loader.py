import torch
import pickle
import json
from torchvision import transforms

from cap_mod_defs import EncoderCNN, DecoderRNN


def load_captioning_model(
    model_path: str,
    vocab_path: str,
    config_path: str,
    device: torch.device = torch.device("cpu")
):
    # -----------------------------
    # Load config
    # -----------------------------
    with open(config_path, "r") as f:
        config = json.load(f)

    max_len = config["max_caption_len"]

    # -----------------------------
    # Load vocab
    # -----------------------------
    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)

    # Handle both vocab formats safely
    if isinstance(vocab_data, dict) and "vocab" in vocab_data:
        vocab = vocab_data["vocab"]
        inv_vocab = vocab_data.get("inv_vocab")
        if inv_vocab is None:
            inv_vocab = {v: k for k, v in vocab.items()}
    else:
        vocab = vocab_data
        inv_vocab = {v: k for k, v in vocab.items()}


    # -----------------------------
    # Load checkpoint
    # -----------------------------
    checkpoint = torch.load(model_path, map_location=device)

    # -----------------------------
    # Recreate models
    # -----------------------------
    encoder = EncoderCNN().to(device)
    decoder = DecoderRNN(vocab_size=checkpoint["vocab_size"]).to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    encoder.eval()
    decoder.eval()

    # -----------------------------
    # Image preprocessing
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -----------------------------
    # Return bundle
    # -----------------------------
    return {
        "encoder": encoder,
        "decoder": decoder,
        "vocab": vocab,
        "inv_vocab": inv_vocab,
        "max_len": max_len,
        "transform": transform
    }
