import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from clearml import Task
import matplotlib.pyplot as plt
from datasets import get_datasets, get_cat_datasets
from losses import categorical_accuracy, CategoricalAccLoss, DummyLoss
from context.test.test import test
from common.models import get_vgg
from config import TrainConfig

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def train_vgg(df: pd.DataFrame, weights_dir: str, config):
    epochs = config.num_epochs
    writer = SummaryWriter(log_dir=weights_dir, comment=config.exp_name)
    category_id_dict = get_category_id_dict_from_df(df)
    train_dataset, val_dataset = get_cat_datasets(df, augment_train = False)
    # test_dataset = TestDataset(config.df_test)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = get_vgg(config.num_classes)
    model.to(device)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_bce  = nn.BCEWithLogitsLoss()
    #criterion = CategoricalAccLoss(category_id_dict)
    #criterion = DummyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
    acc = torchmetrics.Accuracy(task='multiclass', num_classes=config.num_classes).to(device)
    best_loss = 100

    # Train loop
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_train_loss, epoch_train_acc, epoch_cat_train_acc = [], [], []

        for i, data in enumerate(train_dataloader):
            if i>=10:
                break
            inputs = data['image'].to(device)
            labels = data['img_class'].to(device)
            category = data['category'].to(device)

            # convert labels to one-hot encoding
            #one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=config.num_classes).to(device)
            one_hot_labels = torch.zeros(labels.size(0), config.num_classes).to(device)
            for i in range(labels.size(0)):
                one_hot_labels[i, labels[i]//10*10:(labels[i]//10*10)+10] = 1

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())
            reshaped_outputs = outputs.reshape(-1, 10, 10)
            average_category_outputs = torch.mean(reshaped_outputs, dim=2)
            #loss = criterion(outputs, labels)

            thresh_probabilities = torch.argmax(outputs, dim=1)
            loss_ce = criterion_ce(outputs, labels)
            loss_ce_cat = criterion_ce(average_category_outputs, category)
            loss_bce = criterion_bce(outputs, one_hot_labels.float())
            #loss = criterion(outputs, labels)
            loss = 0.5*loss_ce + 0.5*loss_bce
            loss.backward()
            optimizer.step()




            cur_train_acc = acc(thresh_probabilities, labels)
            curr_cat_train_acc = categorical_accuracy(thresh_probabilities, category, category_id_dict)

            # print statistics
            epoch_train_loss.append(loss.item())
            epoch_train_acc.append(cur_train_acc.to(device))
            epoch_cat_train_acc.append(curr_cat_train_acc)
            if i % 10 == 0:  # print every x mini-batches
                print(
                    f'[epoch: {epoch + 1}/{epochs},step: {i + 1:5d}/{len(train_dataloader)}] loss: {np.mean(epoch_train_loss):.3f},'
                    f' acc: {np.mean(epoch_train_acc)}, cat_acc: {np.mean(epoch_cat_train_acc)}')
        scheduler.step()
        writer.add_scalar('Loss/train', np.mean(epoch_train_loss), epoch)
        writer.add_scalar('Accuracy/train', np.mean(np.array(epoch_train_acc)), epoch)
        writer.add_scalar('Categorical Accuracy/train', np.mean(np.array(epoch_cat_train_acc)), epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        rdm_matrix = test(test_dataloader, model, category_id_dict)
        plt.imshow(rdm_matrix)
        plt.title(f'RDM')
        plt.show()
        plt.savefig(os.path.join(f'{weights_dir}/rdm_{epoch}.png'))
        plt.close()

        # logger.current_logger().report_image('RDM', 'test',image=rdm_matrix, iteration= epoch)
        # Validation loop
        model.eval()
        epoch_val_loss, epoch_val_acc, epoch_cat_val_acc = [], [], []
        for i, data in enumerate(val_dataloader):
            if i>=10:
                break
            inputs = data['image'].to(device)
            labels = data['img_class'].to(device)
            category = data['category'].to(device)

            # forward
            outputs = model(inputs.float())

            loss = criterion(outputs, labels)
            thresh_probabilities = torch.argmax(outputs, dim=1)
            cur_val_acc = acc(thresh_probabilities, labels)
            curr_cat_val_acc = categorical_accuracy(thresh_probabilities, category, category_id_dict)

            epoch_val_loss.append(loss.item())
            epoch_val_acc.append(cur_val_acc.to(device))
            epoch_cat_val_acc.append(curr_cat_val_acc)

            if loss < best_loss:
                torch.save(model, f'{weights_dir}/epoch_{epoch}_loss_{loss}')
                best_loss = loss

        print(f'[epoch: {epoch + 1}/{epochs}] loss: {np.mean(epoch_val_loss):.3f}, acc: {np.mean(epoch_val_acc)},  cat_val_acc: {np.mean(epoch_cat_val_acc)}')
        writer.add_scalar('Loss/val', np.mean(epoch_val_loss), epoch)
        writer.add_scalar('Accuracy/val', np.mean(np.array(epoch_val_acc)), epoch)
        writer.add_scalar('Categorical Accuracy/val', np.mean(np.array(epoch_cat_val_acc)), epoch)


    print('Finished Training')


def get_category_id_dict_from_df(df: pd.DataFrame):
    category_id_dict = {}
    for ind, row in df.iterrows():
        if row['class'] not in category_id_dict:
            category_id_dict[row['class']] = row['category']
    return category_id_dict


if __name__ == '__main__':
    data_dir = r"C:\Users\shiri\Documents\School\Master\Research\Face_recognition\Data\VGG-Face2\MTCNN\train_mtcnn_aligned_2000"
    results_dir = r"C:\Users\shiri\Documents\School\Master\Research\Context"

    config = TrainConfig()
    num_classes = config.num_classes
    exp_name = config.exp_name
    train_df = pd.read_csv(config.train_df_path)

    exp_dir = f'{results_dir}/exps/{config.exp_name}_{datetime.now().strftime("%d_%m_%Y-%H_%M_%S")}'
    weights_dir = f'{exp_dir}/weights'

    for d in [exp_dir, weights_dir]:
        if not os.path.isdir(d):
            os.mkdir(d)
    current_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    # task = Task.init(project_name="face_recognition", task_name=f"test_{current_time}")
    # logger = task.get_logger()
    train_vgg(train_df, weights_dir, config)

