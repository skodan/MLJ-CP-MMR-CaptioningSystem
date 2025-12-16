import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import pickle
from PIL import Image
from io import BytesIO

class FlickrHFDataset(Dataset):
    def __init__(self, hf_dataset, vocab_path, caption_field="caption_0"):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        self.dataset = hf_dataset
        self.word2int = vocab["word2int"]
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
        example = self.dataset[idx]
        caption = example[self.caption_field].lower().strip()
        tokens = caption.split()
        tokens = ["<sos>"] + tokens + ["<eos>"]
        caption_ids = [self.word2int.get(token, self.word2int["<unk>"]) for token in tokens]

        image_dict = example["image"]
        image = Image.open(BytesIO(image_dict["bytes"])).convert("RGB")
        image_tensor = self.transform(image)

        return image_tensor, torch.tensor(caption_ids)

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions_padded
