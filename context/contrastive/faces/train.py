import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from clearml import Task
from sklearn.decomposition import PCA

from dataset import TripletDatasetWomenMen, TripletDataset, TripletDatasetBase
from contrastive.losses import ContrastiveLoss
from config import TrainConfig
from contrastive.model import SiameseNetwork
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)

contrastive_model_output_size = 128


def train_contrastive(config, logger):
    print('training contrastive')
    epochs = config.num_epoch_contrastive

    train_dataset = TripletDataset(config.data_dir, config.num_classes, config.num_samples_per_class,
                                   split_to_categories=config.split_to_categories, num_categories=config.num_categories)

    train_dataset, val_dataset = train_dataset.split_to_subset(0.8)
    val_dataset, test_dataset = val_dataset.split_to_subset(0.8)
    category_mapping = train_dataset.class_to_category


    # Create a DataLoader for training
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    model = SiameseNetwork(embedding_dim=contrastive_model_output_size)

    model.to(device)
    #criterion = ContrastiveLoss(temperature=0.5)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1)
    #get_tsne_of_representations(model, test_dataloader, logger, 0)
    # Train loop
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_train_loss = []

        for batch_idx, (anchor, positive, negative, _) in enumerate(train_dataloader):
            if batch_idx > 10:
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
                             series='Train', value=np.mean(epoch_train_loss), iteration=epoch)
        #get_tsne_of_representations(model, test_dataloader, logger, epoch)
        scheduler.step()

        model.eval()
        epoch_val_loss = []

        for batch_idx, (anchor, positive, negative, _) in enumerate(val_dataloader):
            if batch_idx > 10:
                break
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            output_anchor, output_positive, output_negative = model(anchor.float(), positive.float(), negative.float())
            loss = criterion(output_anchor, output_positive, output_negative)
            epoch_val_loss.append(loss.item())


        print (f'Validation loss: {np.mean(epoch_val_loss):.3f}')
        logger.report_scalar(title='Contrastive Loss'.format(epoch),
                             series='Val', value=np.mean(epoch_val_loss), iteration=epoch)

    #save model
    torch.save(model.state_dict(), 'contrastive_model.pth')
    #load the model
    model.load_state_dict(torch.load('contrastive_model.pth'))
    return model, train_dataloader, val_dataloader, test_dataloader, category_mapping


def train_classifier(config, logger):

    train_dataset = TripletDataset(config.data_dir, config.num_classes, config.num_samples_per_class,
                                   config.split_to_categories, config.num_categories)

    # split the dataset in train and test set
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) * 0.8),
                                                                                len(train_dataset) - int(
                                                                                    len(train_dataset) * 0.8)])

    #split val_dataset to val and test
    val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [int(len(val_dataset) * 0.8),
                                                                                len(val_dataset) - int(
                                                                                    len(val_dataset) * 0.8)])

    # Create a DataLoader for training
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    model =torch.hub.load('pytorch/vision:v0.10.0', 'vgg16').eval()
    model.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=10)

    model.to(device)
    #criterion = ContrastiveLoss(temperature=0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1)
    #get_tsne_of_representations(model, test_dataloader, logger, 0)
    # Train loop
    num_epochs = 100
    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_train_loss = []

        for batch_idx, (anchor, _, _, label) in enumerate(train_dataloader):
            inputs, label = anchor.to(device), label.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, label.type(torch.LongTensor))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item())

            # print loss
            if batch_idx % 10 == 0:  # print every x mini-batches
                print(
                    f'[epoch: {epoch + 1}/{num_epochs},step: {batch_idx + 1:5d}/{len(val_dataloader)}] , loss: {np.mean(epoch_train_loss):.3f},')

        logger.report_scalar(title='Transfer Loss', series='Train', value=np.mean(epoch_train_loss), iteration=epoch)

        model.eval()
        epoch_val_loss = []
        for batch_idx, (anchor, _, _, label) in enumerate(val_dataloader):
            inputs, label = anchor.to(device), label.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, label.type(torch.LongTensor))
            epoch_val_loss.append(loss.item())

        print(f'Validation loss: {np.mean(epoch_val_loss):.3f}')
        logger.report_scalar(title='Transfer Loss', series='Val', value=np.mean(epoch_val_loss), iteration=epoch)

        scheduler.step()

    with torch.no_grad():
        accuracy = []
        for batch_idx, (anchor, _, _, label) in enumerate(test_dataloader):
            inputs, labels = anchor.to(device), label.to(device)
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs.data, 1)
            # Compute accuracy
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy.append(correct / total)

    logger.report_text(f'Accuracy of the network on the test set: {np.mean(accuracy):.3f}')


def transfer_model(model, train_dataloader, val_dataloader, test_dataloader, category_mapping, config, logger):
    print('training transfer model')
    feature_extractor = nn.Sequential(*list(model.children()))
    fine_tune_model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, config.num_classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(fine_tune_model.parameters(), lr=0.001, momentum=0.9)


    num_epochs = config.num_epoch_contrastive
    for epoch in range(num_epochs):
        epoch_loss = []
        fine_tune_model.train()
        for batch_idx, (anchor, _, _, label) in enumerate(train_dataloader):
            if batch_idx >2:
                break
            inputs, label = anchor.to(device), label.to(device)

            # Pass inputs through the feature extractor
            features = feature_extractor(inputs)
            features = features.view(features.size(0), -1)

            # Forward pass
            outputs = fine_tune_model(features)
            loss = criterion(outputs, label.type(torch.LongTensor).to(device))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

            # print loss
            if batch_idx % 10 == 0:  # print every x mini-batches
                print(
                    f'[epoch: {epoch + 1}/{num_epochs},step: {batch_idx + 1:5d}/{len(train_dataloader)}] , loss: {np.mean(epoch_loss):.3f},')

        logger.report_scalar(title='Transfer Loss', series='Train', value=np.mean(epoch_loss), iteration=epoch)

        fine_tune_model.eval()
        epoch_val_loss = []
        for batch_idx, (anchor, _, _, label) in enumerate(val_dataloader):
            inputs, label = anchor.to(device), label.to(device)

            # Pass inputs through the feature extractor
            features = feature_extractor(inputs)
            features = features.view(features.size(0), -1)

            # Forward pass
            outputs = fine_tune_model(features)
            loss = criterion(outputs, label.type(torch.LongTensor).to(device))
            epoch_val_loss.append(loss.item())

        print(f'Validation loss: {np.mean(epoch_val_loss):.3f}')
        logger.report_scalar(title='Transfer Loss', series='Val', value=np.mean(epoch_val_loss), iteration=epoch)


    # Test the fine-tuned model
    fine_tune_model.eval()
    with torch.no_grad():
        accuracy = []
        category_accuracy = []
        for batch_idx, (anchor, _, _, label) in enumerate(test_dataloader):
            inputs, labels = anchor.to(device), label.to(device)
            category = [category_mapping[label.item()] for label in labels]
            features = feature_extractor(inputs)
            features = features.view(features.size(0), -1)

            outputs = fine_tune_model(features)
            _, predicted = torch.max(outputs.data, 1)
            # Compute accuracy
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy.append(correct / total)


            predicted_category = [category_mapping[pred.item()] for pred in predicted]
            correct_category = sum([1 if pred == cat else 0 for pred, cat in zip(predicted_category, category)])
            category_accuracy.append(correct_category / total)


    logger.report_text(f'Accuracy of the network on the test set: {np.mean(accuracy):.3f},'
                       f'Category Accuracy: {np.mean(category_accuracy):.3f}')

    return fine_tune_model


def get_tsne_of_representations(model, test_dataloader, logger, epoch):
    feature_extractor = nn.Sequential(*list(model.children()))
    feature_extractor.eval()
    with torch.no_grad():
        features, labels, categories = [], [], []
        for batch_idx, (anchor, _, _, label) in enumerate(test_dataloader):
            inputs, label = anchor.to(device), label.to(device)
            category = [test_dataloader.dataset.target_to_category[label.item()] for label in label]
            features.append(feature_extractor(inputs).cpu().numpy())
            labels.append(label.cpu().numpy())
            categories.append(category)

    features = np.concatenate(features)
    labels = np.concatenate(labels)
    categories = np.concatenate(categories)

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
        palette=sns.color_palette("hls", len(set(categories))),
    )
    logger.report_matplotlib_figure(
        title="TSNE", series="plot", iteration=epoch, figure=plt, report_image=True
    )
    return tsne_results, labels

if __name__ == '__main__':

    config = TrainConfig()
    exp_name = config.exp_name

    current_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    task = Task.init(project_name="faces_triplets", task_name=f"test_{current_time}")
    logger = task.get_logger()
    model, train_dataloader, val_dataloader, test_dataloader, category_mapping = train_contrastive(config, logger)
    fine_tune_model = transfer_model(model,train_dataloader, val_dataloader, test_dataloader, category_mapping, config, logger)
