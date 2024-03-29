import os
import random

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from clearml import Task
from sklearn.decomposition import PCA

from dataset import TripletCIFAR10Dataset
from contrastive.losses import ContrastiveLoss
from config import TrainConfig
from contrastive.model import SiameseNetwork
from sklearn.cluster import KMeans

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

contrastive_model_output_size = 128


def train_contrastive(config, logger):
    print('training contrastive')
    epochs = config.num_epoch_contrastive
    num_classes = config.num_classes
    # Create the triplet dataset
    cifar10_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    train_dataset = TripletCIFAR10Dataset(cifar10_train_dataset, num_classes)

    cifar10_val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    if num_classes !=10:
        indices_list = np.array(range(num_classes))
        class_indices = np.where(np.isin(cifar10_val_dataset.targets, indices_list))[0]
        cifar10_val_dataset.data = cifar10_val_dataset.data[class_indices]
        cifar10_val_dataset.targets = np.array(cifar10_val_dataset.targets)[class_indices]

    # split validation dataset into validation and test
    cifar10_val_dataset, cifar10_test_dataset = torch.utils.data.random_split(cifar10_val_dataset,
                                                                              [int(0.6 * len(cifar10_val_dataset)),
                                                                               int(0.4 * len(cifar10_val_dataset))])



    # Create a DataLoader for training
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(cifar10_val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(cifar10_test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    model = SiameseNetwork(embedding_dim=contrastive_model_output_size)

    model.to(device)
    criterion = ContrastiveLoss(temperature=0.5)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1)
    get_tsne_of_representations(model, test_dataloader, logger, 0)
    # Train loop
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_train_loss = []

        for batch_idx, (anchor, positive, negative) in enumerate(train_dataloader):
            if batch_idx > 2:
                break
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            output_anchor, output_positive, output_negative = model(anchor.float(), positive.float(), negative.float())
            loss = criterion(output_anchor, output_positive, output_negative)
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item())

            if batch_idx % 10 == 0:  # print every x mini-batches
                print(f'[epoch: {epoch + 1}/{epochs},step: {batch_idx + 1:5d}/{len(train_dataloader)}] '
                      f'loss: {np.mean(epoch_train_loss):.3f},')

        logger.report_scalar(title='Contrastive'.format(epoch),
                             series='Loss', value=np.mean(epoch_train_loss), iteration=epoch)
        get_tsne_of_representations(model, test_dataloader, logger, epoch)
    return model, val_dataloader, test_dataloader


def transfer_model(model, val_dataloader, test_dataloader, config, logger):
    print('training transfer model')
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    fine_tune_model = nn.Sequential(
        nn.Linear(1000, 256),
        nn.ReLU(),
        nn.Linear(256, 10)  # Assuming 10 classes for CIFAR-10
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(fine_tune_model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = config.num_epoch_contrastive
    for epoch in range(num_epochs):
        epoch_loss = []
        fine_tune_model.train()
        for batch_idx, (inputs, labels) in enumerate(val_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Pass inputs through the feature extractor
            features = feature_extractor(inputs)
            features = features.view(features.size(0), -1)

            # Forward pass
            outputs = fine_tune_model(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

            # print loss
            if batch_idx % 10 == 0:  # print every x mini-batches
                print(
                    f'[epoch: {epoch + 1}/{num_epochs},step: {batch_idx + 1:5d}/{len(val_dataloader)}] , loss: {np.mean(epoch_loss):.3f},')

        logger.report_scalar(title='Transfer', series='Loss', value=np.mean(epoch_loss), iteration=epoch)


    # Test the fine-tuned model
    fine_tune_model.eval()
    with torch.no_grad():
        accuracy = []
        for batch_idx, (inputs, labels) in enumerate(test_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            features = feature_extractor(inputs)
            features = features.view(features.size(0), -1)

            outputs = fine_tune_model(features)
            _, predicted = torch.max(outputs.data, 1)
            # Compute accuracy
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy.append(correct / total)

    logger.report_text(f'Accuracy of the network on the test set: {np.mean(accuracy):.3f}')

    return fine_tune_model


def get_tsne_of_representations(model, test_dataloader, logger, epoch):
    feature_extractor = nn.Sequential(*list(model.children()))
    feature_extractor.eval()
    with torch.no_grad():
        features = []
        labels = []
        for batch_idx, (inputs, label) in enumerate(test_dataloader):
            inputs, label = inputs.to(device), label.to(device)
            features.append(feature_extractor(inputs).cpu().numpy())
            labels.append(label.cpu().numpy())

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    #plot

    # pca = PCA(n_components=50)
    # pca.fit(features)
    # pca_features = pca.transform(features)

    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(features)
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=labels,
        palette=sns.color_palette("hls", len(labels)),
        legend="full",
        alpha=0.3
    )
    plt.show()
    logger.report_matplotlib_figure(
        title="TSNE", series="plot", iteration=epoch, figure=plt, report_image=True
    )
    return tsne_results, labels

if __name__ == '__main__':
    results_dir = r"C:\Users\shiri\Documents\School\Master\Research\Context"

    config = TrainConfig()
    exp_name = config.exp_name

    exp_dir = f'{results_dir}/exps/{config.exp_name}_{datetime.now().strftime("%d_%m_%Y-%H_%M_%S")}'
    weights_dir = f'{exp_dir}/weights'

    for d in [exp_dir, weights_dir]:
        if not os.path.isdir(d):
            os.mkdir(d)
    current_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    task = Task.init(project_name="face_recognition", task_name=f"test_{current_time}")
    logger = task.get_logger()
    #writer = SummaryWriter(log_dir=weights_dir, comment=config.exp_name)
    model, val_dataloader, test_dataloader = train_contrastive(config, logger)
    fine_tune_model = transfer_model(model, val_dataloader, test_dataloader, config, logger)
