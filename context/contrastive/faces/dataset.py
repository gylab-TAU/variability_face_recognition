import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import numpy as np
from PIL import Image

from utils import Rescale

random.seed(42)
transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
class TripletDataset(Dataset):
    def __init__(self, folder, num_classes = None, num_samples = None):
        self.dataset = self.create_dataset_from_folder(folder)
        if num_classes is None:
            self.num_classes = len(set(self.dataset.targets))
        else:
            self.num_classes = num_classes
        self.num_samples = num_samples
        self._create_subset()
        self.class_to_indices = self._create_class_to_indices()


    def create_dataset_from_folder(self, folder):
        dataset = torchvision.datasets.ImageFolder(
            root=folder,
            transform=torchvision.transforms.ToTensor()
        )
        return dataset

    def _create_subset(self):
        chosen_classes = np.array(range(self.num_classes))
        chosen_indices = []
        for i in range(len(chosen_classes)):
            class_inidices = np.where(np.isin(self.dataset.targets, chosen_classes[i]))[0]
            chosen_indices.extend(class_inidices[:self.num_samples])
        chosen_classes_indices = np.array(chosen_indices)

        self.dataset.imgs = [self.dataset.imgs[i][0] for i in chosen_classes_indices]
        self.dataset.targets = np.array(self.dataset.targets)[chosen_classes_indices]

    def _create_class_to_indices(self):
        class_to_indices = {}
        for i, target in enumerate(self.dataset.targets):
            if target not in class_to_indices:
                class_to_indices[target] = []
            class_to_indices[target].append(i)
        return class_to_indices

    def _get_random_target(self, current_target):
        possible_targets = list(range(self.num_classes))
        possible_targets.remove(current_target)
        return random.choice(possible_targets)

    def _get_triplet(self, anchor_target):
        positive_index = random.choice(self.class_to_indices[anchor_target])
        negative_target = self._get_random_target(anchor_target)
        negative_index = random.choice(self.class_to_indices[negative_target])
        return positive_index, negative_index

    def __len__(self):
        return len(self.dataset.imgs)

    def __getitem__(self, idx):
        anchor_index = idx
        anchor_target = self.dataset.targets[idx]

        positive_index, negative_index = self._get_triplet(anchor_target)

        anchor_image = Image.open(self.dataset.imgs[anchor_index])
        positive_image = Image.open(self.dataset.imgs[positive_index])
        negative_image = Image.open(self.dataset.imgs[negative_index])

        #load images

        anchor_image = transform(anchor_image)
        positive_image = transform(positive_image)
        negative_image = transform(negative_image)

        return anchor_image, positive_image, negative_image, anchor_target



if __name__=='__main__':
    path = r'C:\Users\shiri\Documents\School\Master\Research\Face_recognition\Data\VGG-Face2\MTCNN\train_mtcnn_aligned_2000'
    dataset = TripletDataset(path,num_classes=20, num_samples=10)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    for i, data in enumerate(dataloader):
        anchor, positive, negative = data
        print(anchor.shape)




