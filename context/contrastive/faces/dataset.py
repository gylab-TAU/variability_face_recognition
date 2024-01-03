import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import numpy as np
from PIL import Image
import albumentations as A
from utils import Rescale
import copy

random.seed(42)
augs = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])
transform = transforms.Compose([transforms.ToTensor(), Rescale((64, 64)),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])


class TripletDatasetBase(Dataset):
    def __init__(self, dataset, augs=False):
        self.dataset = self.create_dataset_from_folder(dataset)
        self.num_classes = None
        self.num_samples = None
        self.num_categories = None
        self.split_to_categories = None
        self.class_to_indices = None
        self.augs = augs

    def create_dataset_from_folder(self, folder):
        dataset = torchvision.datasets.ImageFolder(
            root=folder,
            transform=torchvision.transforms.ToTensor()
        )

        return dataset

    def _create_subset(self):
        self.chosen_classes = list(range(self.num_classes))
        chosen_indices = []
        for i in range(len(self.chosen_classes)):
            class_inidices = np.where(np.isin(self.dataset.targets, self.chosen_classes[i]))[0]
            chosen_indices.extend(class_inidices[:self.num_samples])
        chosen_classes_indices = np.array(chosen_indices)

        self.dataset.imgs = [self.dataset.imgs[i][0] for i in chosen_classes_indices]
        self.dataset.targets = np.array(self.dataset.targets)[chosen_classes_indices]

    def _shuffle_dataset(self):
        indices = list(range(len(self.dataset.imgs)))
        random.shuffle(indices)
        self.dataset.imgs = [self.dataset.imgs[i] for i in indices]
        self.dataset.targets = [self.dataset.targets[i] for i in indices]

    def _create_class_to_indices(self):
        self.class_to_indices = {}
        for i, target in enumerate(self.dataset.targets):
            if target not in self.class_to_indices:
                self.class_to_indices[target] = []
            self.class_to_indices[target].append(i)

    def _get_random_target(self, current_target):
        possible_targets = self.chosen_classes.copy()
        possible_targets.remove(current_target)
        return random.choice(possible_targets)

    def _get_random_category(self, current_category):
        possible_categories = list(range(self.num_categories))
        possible_categories.remove(current_category)
        return random.choice(possible_categories)

    def _get_triplet(self, anchor_target):
        positive_index = random.choice(self.class_to_indices[anchor_target])
        negative_target = self._get_random_target(anchor_target)
        negative_index = random.choice(self.class_to_indices[negative_target])
        return positive_index, negative_index

    def _get_triplet_with_category(self, anchor_target):
        anchor_category = self.target_to_category[anchor_target]
        positive_category = anchor_category
        positive_class = random.choice(self.categories_to_classes[positive_category])
        positive_index = random.choice(self.class_to_indices[positive_class])
        negative_category = self._get_random_category(anchor_category)
        negative_class = random.choice(self.categories_to_classes[negative_category])
        negative_index = random.choice(self.class_to_indices[negative_class])
        return positive_index, negative_index

    def _get_triplet_positive_class_negative_category(self, anchor_target):
        positive_index = random.choice(self.class_to_indices[anchor_target])
        anchor_category = self.target_to_category[anchor_target]
        negative_category = self._get_random_category(anchor_category)
        negative_class = random.choice(self.categories_to_classes[negative_category])
        negative_index = random.choice(self.class_to_indices[negative_class])
        return positive_index, negative_index

    def _split_targets_to_categories(self):
        #split to categories
        categories_to_classes = {}
        classes = list(range(self.num_classes))
        random.shuffle(classes)
        for i in range(self.num_categories):
            categories_to_classes[i] = classes[i * self.num_classes // self.num_categories:(i + 1) * self.num_classes // self.num_categories]
        self.categories_to_classes = categories_to_classes

        # create dict of classes to categories
        class_to_category = {}
        for category in categories_to_classes:
            for class_ in categories_to_classes[category]:
                class_to_category[class_] = category
        self.class_to_category = class_to_category

        # create a list of target to category
        targets_to_category = []
        for target in self.dataset.targets:
            targets_to_category.append(class_to_category[target])
        self.target_to_category = targets_to_category

    def split_to_subset(self, train_size=0.8):
        train_dataset = copy.deepcopy(self)
        val_dataset = copy.deepcopy(self)
        train_indices = random.sample(range(len(self.dataset.imgs)), int(train_size * len(self.dataset.imgs)))
        val_indices = list(set(range(len(self.dataset.imgs))) - set(train_indices))
        train_dataset.dataset.imgs = [self.dataset.imgs[i] for i in train_indices]
        train_dataset.dataset.targets = [self.dataset.targets[i] for i in train_indices]
        val_dataset.dataset.imgs = [self.dataset.imgs[i] for i in val_indices]
        val_dataset.dataset.targets = [self.dataset.targets[i] for i in val_indices]
        train_dataset._create_class_to_indices()
        train_dataset._split_targets_to_categories()
        val_dataset._create_class_to_indices()
        val_dataset._split_targets_to_categories()
        return train_dataset, val_dataset

    def __len__(self):
        return len(self.dataset.imgs)

    def __getitem__(self, idx):
        anchor_index = idx
        anchor_target = self.dataset.targets[idx]

        if self.split_to_categories:
            p = random.random()
            if p < 0.5:
                positive_index, negative_index = self._get_triplet(anchor_target)
            else:
                positive_index, negative_index = self._get_triplet_positive_class_negative_category(anchor_target)
        else:
            positive_index, negative_index = self._get_triplet(anchor_target)

        anchor_image = Image.open(self.dataset.imgs[anchor_index])
        positive_image = Image.open(self.dataset.imgs[positive_index])
        negative_image = Image.open(self.dataset.imgs[negative_index])

        if self.augs is True:
            anchor_image = augs(image=np.array(anchor_image))["image"]
            positive_image = augs(image=np.array(positive_image))["image"]
            negative_image = augs(image=np.array(negative_image))["image"]

        anchor_image = transform(anchor_image)
        positive_image = transform(positive_image)
        negative_image = transform(negative_image)

        return anchor_image, positive_image, negative_image, anchor_target


class TripletDataset(TripletDatasetBase):
    def __init__(self, dataset, num_classes=None, num_samples=10, split_to_categories=False, num_categories=2,
                 augs=False):
        super().__init__(dataset)
        self.split_to_categories = split_to_categories
        self.num_categories = num_categories
        self.dataset = self.create_dataset_from_folder(dataset)
        if num_classes is None:
            self.num_classes = len(set(self.dataset.targets))
        else:
            self.num_classes = num_classes
        self.num_samples = num_samples
        self._create_subset()
        self._shuffle_dataset()
        self._create_class_to_indices()
        if self.split_to_categories:
            self._split_targets_to_categories()
        self.augs = augs


class TripletDatasetWomenMen(TripletDatasetBase):
    def __init__(self, dataset, num_samples=10):
        super().__init__(dataset)
        self.num_samples = num_samples
        self.num_categories = 2
        self._create_subset()
        self._shuffle_dataset()
        self._create_class_to_indices()
        self._split_targets_to_categories()
        self.augs = None

    def _create_subset(self):
        #self.chosen_classes = [0, 1, 3, 4, 6, 7, 9, 10, 11, 12, 14]
        self.chosen_classes =  [ 0, 1, 3, 4, 6, 7, 8, 13, 17, 18, 19, 20, 21, 23, 24,25, 26,29, 34,36,
                               9,10,11,12,14, 15, 16, 22, 27, 28, 30, 31,32,33, 35,37, 38, 41,43, 49]
        chosen_indices = []
        for i in range(len(self.chosen_classes)):
            class_inidices = np.where(np.isin(self.dataset.targets, self.chosen_classes[i]))[0]
            chosen_indices.extend(class_inidices[:self.num_samples])
        chosen_classes_indices = np.array(chosen_indices)

        self.dataset.imgs = [self.dataset.imgs[i][0] for i in chosen_classes_indices]
        self.dataset.targets = np.array(self.dataset.targets)[chosen_classes_indices]
        self.mapping = {self.chosen_classes[i]: i for i in range(len(self.chosen_classes))}
        self.dataset.targets = np.array([self.mapping[i] for i in self.dataset.targets])
        self.chosen_classes = list(range(len(self.chosen_classes)))

    def _split_targets_to_categories(self):
        self.categories_to_classes = {0: list(range(20)), 1: list(range(20,40))}
        class_to_category = {}
        for category in self.categories_to_classes:
            for class_ in self.categories_to_classes[category]:
                class_to_category[class_] = category
        self.class_to_category = class_to_category
        targets_to_category = []
        for target in self.dataset.targets:
            targets_to_category.append(class_to_category[target])
        self.target_to_category = targets_to_category



    def __getitem__(self, idx):
        anchor_index = idx
        anchor_target = self.dataset.targets[idx]

        p = random.random()
        if p < 0.5:
            positive_index, negative_index = self._get_triplet(anchor_target)
        else:
            positive_index, negative_index = self._get_triplet_with_category(anchor_target)

        anchor_image = Image.open(self.dataset.imgs[anchor_index])
        positive_image = Image.open(self.dataset.imgs[positive_index])
        negative_image = Image.open(self.dataset.imgs[negative_index])

        anchor_image = transform(anchor_image)
        positive_image = transform(positive_image)
        negative_image = transform(negative_image)

        return anchor_image, positive_image, negative_image, anchor_target


if __name__ == '__main__':
    path = r'C:\Users\shiri\Documents\School\Master\Research\Face_recognition\Data\VGG-Face2\MTCNN\train_mtcnn_aligned_2000'
    dataset = TripletDataset(path, num_classes=20, num_samples=10)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    for i, data in enumerate(dataloader):
        anchor, positive, negative = data
        print(anchor.shape)
