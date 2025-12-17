import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from io import BytesIO
import pickle

class FlickrDataset(Dataset):
    def __init__(self, dataset, vocab_path, image_dir=None, mode="file",transform=None,caption_field="caption_0"):
        self.data = dataset
        self.image_dir = image_dir
        self.caption_field = caption_field
        self.mode = mode

        # with open(vocab_path, "rb") as f:
        #     vocab = pickle.load(f)
        # self.vocab = vocab["word2int"]

        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        # Support BOTH CSV (tuple) and HF (dict) vocab formats
        if isinstance(vocab, tuple):
            self.vocab = vocab[0]          # word2int
        elif isinstance(vocab, dict):
            self.vocab = vocab["word2int"]
        else:
            raise ValueError("Unsupported vocab format")

        self.transform = transform or transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if self.mode == "file":
            row = self.data.iloc[idx]  # CSV → DataFrame
        else:
            row = self.data[idx]

        if self.mode == "file":
            image = Image.open(os.path.join(self.image_dir, row['image'])).convert('RGB')
        else:  # HF dataset
            image = row['image']

        image = self.transform(image)
        caption = row[self.caption_field].lower().strip()
        #caption = row['caption'].lower()
        tokens = caption.lower().strip().split()
        caption_ids = [self.vocab['<sos>']] + [self.vocab.get(token, self.vocab['<unk>']) for token in tokens] + [self.vocab['<eos>']]
        #caption_ids = [self.vocab['<bos>']] + [self.vocab.get(token, self.vocab['<unk>']) for token in tokens] + [self.vocab['<eos>']]
        #caption_ids = [self.vocab['<BOS>']] + [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens] + [self.vocab['<EOS>']]
        
        return image, torch.tensor(caption_ids), row['image']


def collate_fn(batch):
    images, captions, _ = zip(*batch)
    lengths = [len(cap) for cap in captions]
    padded = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
    return torch.stack(images), padded, lengths