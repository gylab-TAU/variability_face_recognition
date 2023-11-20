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

from dataset import get_datasets
from losses import ContrastiveLoss
from config import TrainConfig
from model import SiameseNetwork

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def train_contrastive(df: pd.DataFrame, weights_dir: str, config):
    epochs = config.num_epochs
    writer = SummaryWriter(log_dir=weights_dir, comment=config.exp_name)

    train_dataset, val_dataset = get_datasets(df)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = SiameseNetwork()

    model.to(device)
    criterion = ContrastiveLoss(temperature=0.5)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    # Train loop
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_train_loss = []

        for batch_idx, (anchor, positive, negative) in enumerate(train_dataloader):
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

        # rdm_matrix = test(test_dataloader, model, category_id_dict)
        # plt.imshow(rdm_matrix)
        # plt.title(f'RDM')
        # plt.show()
        # plt.savefig(os.path.join(f'{weights_dir}/rdm_{epoch}.png'))
        # plt.close()

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
    train_contrastive(train_df, weights_dir, config)