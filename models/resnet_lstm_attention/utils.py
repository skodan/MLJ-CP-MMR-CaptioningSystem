# backend/utils.py
import re

def simple_tokenize(text: str) -> list[str]:
    """
    Same simple tokenizer used during training
    """
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()