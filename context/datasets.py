import pandas as pd
import torch
import torchvision.datasets
from torch.utils.data import Dataset
from torchvision import transforms

from common.mytransforms import Rescale, normalize_imagenet

composed_transforms = transforms.Compose([Rescale((160, 160)), normalize_imagenet])

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VGGDataset(Dataset):

    def __init__(self, df, rescale_size=(160, 160)):
        self.df = df.reset_index()
        self.rescale_size = rescale_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df.loc[idx, 'image_path']
        image = torchvision.io.read_image(img_name)
        image = composed_transforms(image)
        img_class = self.df.loc[idx, 'class']
        sample = {'image': image, 'img_class': img_class}
        return sample


class TestDataset(Dataset):
    def __init__(self, df_path, rescale_size=(160, 160)):
        self.df = pd.read_csv(df_path).reset_index()
        self.rescale_size = rescale_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image1 = composed_transforms(torchvision.io.read_image(self.df.loc[idx, '1']))
        image2 = composed_transforms(torchvision.io.read_image(self.df.loc[idx, '2']))
        sample = {'image1': image1, 'image2': image2, 'label': self.df.loc[idx, 'label']}
        return sample


def get_datasets(df):
    train_dataset = VGGDataset(df[df['set'] == 'train'])
    val_dataset = VGGDataset(df[df['set'] == 'val'])

    return train_dataset, val_dataset
