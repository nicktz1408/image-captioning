import os
import io
import torch
import pandas as pd
import numpy as np
import PIL
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split


class ImageCaptionDataset(Dataset):
    def __init__(self, captions_df, captions_tokenized, root_dir, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_captions_data = captions_df
        self.captions_tokenized = captions_tokenized
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.captions_tokenized)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        caption = self.captions_tokenized[idx]

        img_path = os.path.join(self.root_dir,
                                self.image_captions_data.iloc[idx, 0])

        image = PIL.Image.open(img_path)
        #image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)

        return [ image, caption ]

def create_loaders(captions_df, captions_idx, image_path):
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])

    caption_dset = ImageCaptionDataset(captions_df, captions_idx, root_dir=image_path, transform=image_transforms)
    train_dset, test_dset = random_split(caption_dset, [0.8, 0.2])

    train_loader = DataLoader(train_dset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dset, batch_size=32, shuffle=True, num_workers=0)

    return train_loader, test_loader

if __name__ == 'main':

    dataset_path = '/kaggle/input/flickr8k'

    captions_df = pd.load_csv(dataset_path + '/captions.txt')
    image_path = dataset_path + '/Images'
    captions_idx = np.loadtxt('./captions_tokenized_data.txt')


    train_loader, test_loader = create_loaders(captions_df, captions_idx, image_path)