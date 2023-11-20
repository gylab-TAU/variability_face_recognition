import os
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

from dataset import TripletCIFAR10Dataset
from contrastive.losses import ContrastiveLoss
from config import TrainConfig
from contrastive.model import SiameseNetwork

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

contrastive_model_output_size = 128
def train_contrastive(weights_dir: str, config):
    epochs = config.num_epoch_contrastive
    writer = SummaryWriter(log_dir=weights_dir, comment=config.exp_name)


    # Create the triplet dataset
    cifar10_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    cifar10_val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    # split validation dataset into validation and test
    cifar10_val_dataset, cifar10_test_dataset = torch.utils.data.random_split(cifar10_val_dataset, [int(0.5 * len(cifar10_val_dataset)), int(0.5 * len(cifar10_val_dataset))])

    train_dataset = TripletCIFAR10Dataset(cifar10_train_dataset)


    # Create a DataLoader for training
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(cifar10_val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(cifar10_test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    model = SiameseNetwork()

    model.to(device)
    criterion = ContrastiveLoss(temperature=0.5)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    # Train loop
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_train_loss = []

        for batch_idx, (anchor, positive, negative) in enumerate(train_dataloader):
            if batch_idx > 200:
                break
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            output_anchor, output_positive, output_negative = model(anchor.float(), positive.float(), negative.float())
            loss = criterion(output_anchor, output_positive, output_negative)
            loss.backward()
            optimizer.step()

            # print statistics
            epoch_train_loss.append(loss.item())

            if batch_idx % 10 == 0:  # print every x mini-batches
                print(f'[epoch: {epoch + 1}/{epochs},step: {batch_idx  + 1:5d}/{len(train_dataloader)}] '
                      f'loss: {np.mean(epoch_train_loss):.3f},')


        writer.add_scalar('Loss/train', np.mean(epoch_train_loss), epoch)

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # # Validation loop
        # model.eval()
        # epoch_val_loss = []
        # with torch.no_grad():
        #     for batch_idx, (anchor, positive, negative) in enumerate(val_dataloader):
        #         anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        #         output_anchor, output_positive, output_negative = model(anchor.float(), positive.float(), negative.float())
        #         loss = criterion(output_anchor, output_positive, output_negative)
        #         epoch_val_loss.append(loss.item())
        #
        # writer.add_scalar('Loss/val', np.mean(epoch_val_loss), epoch)

    return model, val_dataloader, test_dataloader


def transfer_model(model, val_dataloader, test_dataloader, config):
    # Assuming model is your pre-trained contrastive model
    # You can create a feature extractor using the layers before the contrastive head
    feature_extractor = nn.Sequential(*list(model.children())[:-1])

    #Fine-tuning for CIFAR-10 classification
    fine_tune_model = nn.Sequential(
        nn.Linear(contrastive_model_output_size*8*8, 256),
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

            #print loss
            if batch_idx % 10 == 0:  # print every x mini-batches
                print(f'[epoch: {epoch + 1}/{num_epochs},step: {batch_idx  + 1:5d}/{len(val_dataloader)}] , loss: {np.mean(epoch_loss):.3f},')

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

    print(f'Accuracy of the network on the test set: {np.mean(accuracy):.3f}')

    return fine_tune_model



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
    # task = Task.init(project_name="face_recognition", task_name=f"test_{current_time}")
    # logger = task.get_logger()
    model, val_dataloader, test_dataloader = train_contrastive(weights_dir, config)
    fine_tune_model = transfer_model(model, val_dataloader, test_dataloader)