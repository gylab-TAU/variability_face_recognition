import pandas as pd
import torch
import torchvision.datasets
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
import cv2
import random
from PIL import Image
import numpy as np

from common.mytransforms import Rescale, normalize_imagenet

random.seed(42)
composed_transforms = transforms.Compose([Rescale((160, 160)), normalize_imagenet])
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ContrastiveCategoryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, add_category: bool = False, augment: bool = False):
        self.df = df.reset_index()
        self.augment = augment
        self.add_category = add_category

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        anchor_img_name = self.df.loc[idx, 'image_path']
        anchor_img_category = self.df.loc[idx, 'category']

        positive_idx = self.df[self.df['category'] == anchor_img_category].sample(1).index[0]
        positive_img_name = self.df.loc[positive_idx, 'image_path']

        negative_idx = self.df[self.df['category'] != anchor_img_category].sample(1).index[0]
        negative_img_name = self.df.loc[negative_idx, 'image_path']

        anchor_image = torchvision.io.read_image(anchor_img_name)
        anchor_image = composed_transforms(anchor_image)
        positive_image = torchvision.io.read_image(positive_img_name)
        positive_image = composed_transforms(positive_image)
        negative_image = torchvision.io.read_image(negative_img_name)
        negative_image = composed_transforms(negative_image)


        return anchor_image, positive_image, negative_image


def get_datasets(df):
    train_dataset = ContrastiveCategoryDataset(df[df['set'] == 'train'], add_category=True)
    val_dataset = ContrastiveCategoryDataset(df[df['set'] == 'val'])

    return train_dataset, val_dataset
