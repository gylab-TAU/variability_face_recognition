import pandas as pd
import torch
import torchvision.datasets
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
import random


from common.mytransforms import Rescale, normalize_imagenet

random.seed(42)
aug_transform = A.Compose([
        #A.HorizontalFlip(p=0.5),
        #A.RandomBrightnessContrast(p=1),
        A.GaussNoise(var_limit=(0, 0.1), mean=0.5, p=1),
        #A.RandomCrop(width=120, height=120, p=0.2)
       # A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0.0),
    ])
composed_transforms = transforms.Compose([Rescale((160, 160)), normalize_imagenet])

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VGGDataset(Dataset):

    def __init__(self, df, rescale_size=(160, 160), augment=False):
        self.df = df.reset_index()
        self.rescale_size = rescale_size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df.loc[idx, 'image_path']
        image = torchvision.io.read_image(img_name)
        image = composed_transforms(image)

        if self.augment:
            image = image.permute(1, 2, 0).numpy()
            image = aug_transform(image=image)['image']
            image = transforms.ToTensor()(image)

        # if self.augment:
        #     image = Image.open(img_name)
        #     image = np.array(image)
        #     image = aug_transform(image=image)['image']
        #     image = transforms.ToTensor()(image)
        # else:
        #     image = torchvision.io.read_image(img_name)
        #     image = composed_transforms(image)

        img_class = self.df.loc[idx, 'class']
        sample = {'image': image, 'img_class': img_class}
        return sample


class  CategoryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, add_category: bool = False, augment: bool =False):
        self.df = df.reset_index()
        self.augment = augment
        self.add_category = add_category

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df.loc[idx, 'image_path']
        img_category = self.df.loc[idx, 'category']
        image = torchvision.io.read_image(img_name)
        image = composed_transforms(image)
        if self.add_category:
            # get random probability between 0 and 1
            p = random.random()
            if p > 0.5:
                image = replace_last_row_with_label(image, img_category)

        img_class = self.df.loc[idx, 'class']
        sample = {'image': image, 'img_class': img_class, 'category': img_category}
        return sample


def replace_last_row_with_label(image: torch.Tensor, label: int):
    p = random.random()
    if p > 0.8:
        image[:, -1, :] = label
    else:
        image[:, -1, :] = 100
    return image

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


def get_datasets(df, augment_train = False):
    train_dataset = VGGDataset(df[df['set'] == 'train'], augment=augment_train)
    val_dataset = VGGDataset(df[df['set'] == 'val'])

    return train_dataset, val_dataset

def get_cat_datasets(df, augment_train = False):
    train_dataset = CategoryDataset(df[df['set'] == 'train'], add_category=True, augment=augment_train)
    val_dataset = CategoryDataset(df[df['set'] == 'val'])

    return train_dataset, val_dataset