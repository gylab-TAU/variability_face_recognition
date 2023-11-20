import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import random

random.seed(42)

class TripletCIFAR10Dataset(Dataset):
    def __init__(self, cifar10_dataset):
        self.cifar10_dataset = cifar10_dataset
        self.num_classes = len(set(cifar10_dataset.targets))
        self.class_to_indices = self._create_class_to_indices()

    def _create_class_to_indices(self):
        class_to_indices = {}
        for i, target in enumerate(self.cifar10_dataset.targets):
            if target not in class_to_indices:
                class_to_indices[target] = []
            class_to_indices[target].append(i)
        return class_to_indices

    def _get_random_target(self, current_target):
        possible_targets = list(range(self.num_classes))
        possible_targets.remove(current_target)
        return random.choice(possible_targets)

    def _get_triplet(self, anchor_index, anchor_target):
        positive_index = random.choice(self.class_to_indices[anchor_target])
        negative_target = self._get_random_target(anchor_target)
        negative_index = random.choice(self.class_to_indices[negative_target])
        return positive_index, negative_index

    def __len__(self):
        return len(self.cifar10_dataset)

    def __getitem__(self, idx):
        anchor_index = idx
        anchor_target = self.cifar10_dataset.targets[idx]

        positive_index, negative_index = self._get_triplet(anchor_index, anchor_target)

        anchor_image, _ = self.cifar10_dataset[anchor_index]
        positive_image, _ = self.cifar10_dataset[positive_index]
        negative_image, _ = self.cifar10_dataset[negative_index]

        transform = transforms.Compose([

            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        anchor_image = transform(anchor_image)
        positive_image = transform(positive_image)
        negative_image = transform(negative_image)

        return anchor_image, positive_image, negative_image



