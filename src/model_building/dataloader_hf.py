from random import random
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
import torch.nn as nn   

def tokenize(caption):
    caption = caption.lower().strip()
    caption = re.sub(r"[^\w\s]", "", caption)  # remove punctuation
    return caption.split()


def tokenize_caption(caption, vocab, max_len=25):
    words = [vocab.get(t, vocab['<unk>']) for t in simple_tokenize(caption)]
    if len(words) > max_len - 2:
        words = words[:max_len - 2]
    tokens = [vocab['<start>']] + words + [vocab['<end>']]
    length = len(tokens)
    padded = tokens + [vocab['<pad>']] * (max_len + 2 - length)
    return torch.tensor(padded, dtype=torch.long), length


def simple_tokenize(text):
    # Remove punctuation + multiple spaces, lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()


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


class Flickr30kCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform or DEFAULT_IMAGE_TRANSFORM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = os.path.join(self.image_root, row["filename"])
        image = Image.open(img_path).convert("RGB")
        if isinstance(image, Image.Image):
            image = self.transform(image)

        # Pick ONE raw caption (string)
        captions = ast.literal_eval(row["raw"])
        caption = captions[
            torch.randint(0, len(captions), (1,)).item()
        ]

        return image, caption


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
        for _ in range(5):   # try max 5 times
            row = self.df.iloc[idx]
            img_path = os.path.join(self.image_root, row["filename"])

            if os.path.exists(img_path):
                break

            idx = random.randint(0, len(self.df) - 1)

        # FINAL fallback (guaranteed valid)
        img_path = os.path.join(self.image_root, self.df.iloc[0]["filename"])

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        captions = ast.literal_eval(row["raw"])
        caption = random.choice(captions)

        caption_ids = self.encode(caption)
        return image, caption_ids



    # def __getitem__(self, idx):
    #     row = self.df.iloc[idx]

    #     if not os.path.exists(img_path):
    #         return self.__getitem__(random.randint(0, len(self.data) - 1))

    #     # Load image
    #     img_path = os.path.join(self.image_root, row["filename"])

    #     image = Image.open(img_path).convert("RGB")
    #     if isinstance(image, Image.Image):
    #         image = self.transform(image)

    #     # Pick ONE caption randomly from the 5
    #     captions = ast.literal_eval(row["raw"])
    #     caption = captions[torch.randint(0, len(captions), (1,)).item()]
    #     caption_ids = self.encode_caption(caption)

    #     return image, caption_ids, self.pad_idx


class Flickr8kHF(Dataset):
    def __init__(self, hf_dataset, vocab, transform=None, max_len=25):
        self.hf_dataset = hf_dataset
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        ex = self.hf_dataset[idx]
        img = ex['image'].convert('RGB')

        if self.transform:
            img = self.transform(img)

        # Pick random caption out of 5
        cap_idx = random.randint(0, 4)
        caption = ex[f"caption_{cap_idx}"]
        tokens, length = tokenize_caption(caption, self.vocab, self.max_len)

        return img, tokens, length, idx, caption


class Flickr30kImageDataset(Dataset):
    def __init__(self, df, image_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform or DEFAULT_IMAGE_TRANSFORM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["filename"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image


class Flickr30kEvalDataset(Dataset):
    def __init__(self, df, vocab, image_root, transform=None):
        self.vocab = vocab
        self.image_root = image_root
        self.transform = transform or DEFAULT_IMAGE_TRANSFORM

        self.word2int = vocab["word2int"]
        self.pad_idx = self.word2int["<pad>"]
        self.sos_idx = self.word2int["<sos>"]
        self.eos_idx = self.word2int["<eos>"]
        self.unk_idx = self.word2int["<unk>"]

        self.samples = []

        for img_id, row in enumerate(df.itertuples()):
            captions = ast.literal_eval(row.raw)
            for cap in captions:
                self.samples.append((img_id, row.filename, cap))

    def __len__(self):
        return len(self.samples)

    def encode_caption(self, caption):
        tokens = tokenize(caption)
        ids = [self.sos_idx]
        for tok in tokens:
            ids.append(self.word2int.get(tok, self.unk_idx))
        ids.append(self.eos_idx)
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        img_id, filename, caption = self.samples[idx]

        img_path = os.path.join(self.image_root, filename)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        caption_ids = self.encode_caption(caption)

        return image, caption_ids, img_id, self.pad_idx


def eval_collate_fn(batch):    
    items = list(zip(*batch))
    
    if len(items) == 4:
        # Format: (image, caption, img_id, pad_idx)
        _, captions, img_ids, pad_idxs = items
    elif len(items) == 3:
        # Format: (caption, img_id, pad_idx)
        captions, img_ids, pad_idxs = items
    else:
        # Fallback for unexpected lengths (e.g., 5 items)
        # Assuming captions are always the second to last or specific index
        captions = items[0] if len(items[0][0]) > 1 else items[1]
        img_ids = items[1] if len(items) < 3 else items[2]
        pad_idxs = items[-1]

    pad_idx = pad_idxs[0]
    lengths = torch.LongTensor([len(c) for c in captions])

    # Pad the sequences
    captions_padded = pad_sequence(
        captions, batch_first=True, padding_value=pad_idx
    )

    # Return exactly 3 items to match your loop: for captions, img_ids, lengths in txt_loader
    return captions_padded, img_ids, lengths


def collate_fn(batch):
    images = []
    captions = []
    lengths = []
    img_ids = []

    for sample in batch:
        if len(sample) == 3:
            image, caption, pad_idx = sample
            img_id = None
        else:
            image, caption, img_id, pad_idx = sample

        images.append(image)
        captions.append(caption)
        lengths.append(len(caption))
        img_ids.append(img_id)

    images = torch.stack(images, dim=0)
    captions = nn.utils.rnn.pad_sequence(
        captions, batch_first=True, padding_value=pad_idx
    )

    if img_ids[0] is None:
        return images, captions, lengths
    else:
        return images, captions, img_ids, lengths


def clip_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]  # RAW strings
    lengths = None  # dummy, unused
    return images, captions, lengths


def collate_fn(batch, vocab):
    """
    Custom collate to handle variable-length captions
    batch is list of tuples: (img, tokens, length, idx, caption_str)
    """
    imgs, tokens_list, lens, idxs, cap_strs = zip(*batch)  # unpack all 5 values

    imgs = torch.stack(imgs)  # shape: (batch_size, 3, 224, 224)

    max_l = max(lens)
    caps_padded = torch.full((len(batch), max_l), vocab['<pad>'], dtype=torch.long)

    for i, (tokens, l) in enumerate(zip(tokens_list, lens)):
        caps_padded[i, :l] = tokens[:l]  # copy valid tokens

    return imgs, caps_padded, lens, idxs, cap_strs  # return all 5