import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
# import nltk
# nltk.download('punkt')

class FlickrDataset(Dataset):
    def __init__(self, dataframe, vocab, image_dir, transform=None):
        self.data = dataframe
        self.vocab = vocab
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(os.path.join(self.image_dir, row['image'])).convert('RGB')
        image = self.transform(image)
        caption = row['caption'].lower()
        #tokens = nltk.tokenize.word_tokenize(caption)
        tokens = caption.lower().strip().split()
        caption_ids = [self.vocab['<BOS>']] + [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens] + [self.vocab['<EOS>']]
        return image, torch.tensor(caption_ids), row['image']


def collate_fn(batch):
    images, captions, _ = zip(*batch)
    lengths = [len(cap) for cap in captions]
    padded = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
    return torch.stack(images), padded, lengths