from abc import ABC, abstractmethod
from typing import List, Dict, Any
from PIL import Image

class UnifiedModelInterface(ABC):
    @abstractmethod
    def load(self) -> None:
        """Lazy load all required components (models, indices, vocab, etc.)"""
        pass

    @abstractmethod
    def generate_caption(self, image: Image.Image) -> str:
        pass

    @abstractmethod
    def text_to_image(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Returns [{'image_path': str, 'score': float}, ...]"""
        pass

    @abstractmethod
    def image_to_text(self, image: Image.Image, top_k: int = 5) -> List[str]:
        """Returns list of caption strings"""
        pass

    @abstractmethod
    def image_to_image(self, image: Image.Image, top_k: int = 5) -> List[Dict[str, Any]]:
        pass

    def text_to_text(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError("Text-to-text not supported by this model")