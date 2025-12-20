import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import pickle
from PIL import Image
from io import BytesIO

class FlickrDataset(Dataset):
    def __init__(self, dataset, vocab_path, caption_field="caption_0"):
        with open(vocab_path, "rb") as f:
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
    

def collate_fn(batch):
    images, captions, pad_idxs = zip(*batch)
    pad_idx = pad_idxs[0]
    length = [len(caption) for caption in captions]
    images = torch.stack(images)
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=pad_idx)
    return images, captions_padded, length
