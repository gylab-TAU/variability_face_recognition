from torchvision.transforms import transforms
from data_augs.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_augs.view_generator import ContrastiveLearningViewGenerator



class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, n_views, dataset_name):
        valid_datasets = {'cifar10_train': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),
                          'cifar10_val': lambda: datasets.CIFAR10(self.root_folder, train=False,
                                                                transform=ContrastiveLearningViewGenerator(
                                                                    self.get_simclr_pipeline_transform(32),
                                                                    n_views),
                                                                  download=True)}

        dataset_fn = valid_datasets[dataset_name]
        return dataset_fn()