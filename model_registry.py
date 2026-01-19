from typing import Dict
from utils.interfaces import UnifiedModelInterface

_LOADED_MODELS: Dict[str, UnifiedModelInterface] = {}

def get_model(model_name: str) -> UnifiedModelInterface:
    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]

    if model_name == "resnet_lstm_attention":
        from models.resnet_lstm_attention.model import ResNetLSTMAttentionModel
        model = ResNetLSTMAttentionModel()
    elif model_name == "vit_lstm_attention":
        # Add later: from models.vit_lstm_attention.model import VitLSTMAttentionModel
        # model = VitLSTMAttentionModel()
        raise NotImplementedError("ViT + LSTM Attention not implemented yet")
    elif model_name == "vit_transformer":
        # Add later: from models.vit_transformer.model import VitTransformerModel
        # model = VitTransformerModel()
        raise NotImplementedError("ViT + Transformer not implemented yet")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load()
    _LOADED_MODELS[model_name] = model
    return model