import re
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import pickle
from PIL import Image
from io import BytesIO
import ast
import os
from torchvision import transforms


def tokenize(caption):
    caption = caption.lower().strip()
    caption = re.sub(r"[^\w\s]", "", caption)  # remove punctuation
    return caption.split()

DEFAULT_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class FlickrDataset(Dataset):
    def __init__(self, dataset, vocab, caption_field="caption_0"):
        if isinstance(vocab, str):
            with open(vocab, "rb") as f:
                vocab = pickle.load(f)
        
        self.vocab = vocab
        self.dataset = dataset
        self.word2int = vocab["word2int"]
        self.pad_idx = self.word2int["<pad>"]
        #self.pad_token = self.word2int["<pad>"]
        self.caption_field = caption_field

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):  
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]

        tokens = row[self.caption_field].lower().strip().split()
        tokens = ["<sos>"] + tokens + ["<eos>"]
        caption_ids = [self.word2int.get(token, self.word2int["<unk>"]) for token in tokens]

        image = row["image"]
        #image = Image.open(BytesIO(image_dict["bytes"])).convert("RGB")
        #image_tensor = self.transform(image), self.pad_idx
        image_tensor = self.transform(image)

        return image_tensor, torch.tensor(caption_ids), self.pad_idx


class Flickr30kDataset(Dataset):
    def __init__(self, df, vocab, image_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.image_root = image_root
        self.transform = transform or DEFAULT_IMAGE_TRANSFORM

        self.word2int = vocab["word2int"]
        self.pad_idx = self.word2int["<pad>"]
        self.sos_idx = self.word2int["<sos>"]
        self.eos_idx = self.word2int["<eos>"]
        self.unk_idx = self.word2int["<unk>"]

    def __len__(self):
        return len(self.df)

    def encode_caption(self, caption):
        tokens = tokenize(caption)
        ids = [self.sos_idx]
        for tok in tokens:
            ids.append(self.word2int.get(tok, self.unk_idx))
        ids.append(self.eos_idx)
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = os.path.join(self.image_root, row["filename"])
        image = Image.open(img_path).convert("RGB")
        if isinstance(image, Image.Image):
            image = self.transform(image)

        # Pick ONE caption randomly from the 5
        captions = ast.literal_eval(row["raw"])
        caption = captions[torch.randint(0, len(captions), (1,)).item()]
        caption_ids = self.encode_caption(caption)

        return image, caption_ids, self.pad_idx



def collate_fn(batch):
    images, captions, pad_idxs = zip(*batch)
    pad_idx = pad_idxs[0]
    length = [len(caption) for caption in captions]
    images = torch.stack(images)
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=pad_idx)
    return images, captions_padded, length
