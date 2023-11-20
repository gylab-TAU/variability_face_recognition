import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import torch
import numpy as np

from context.datasets import get_cat_datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def test(dataset: DataLoader, model, category_id_dict: dict):

    # Validation loop
    model.eval()
    rdm = np.zeros((100, 100, 10))
    # only take 10 from each category
    for i, data in enumerate(dataset):
        if i>=100:
            break
        inputs = data['image'].to(device)
        labels = data['img_class'].to(device)
        label_category = data['category'].to(device)

        # forward
        outputs = model(inputs.float())
        probabilities = F.softmax(outputs, dim=1)

        thresh_probabilities = torch.argmax(outputs, dim=1)
        rdm[labels, :, i%10] = probabilities.detach().cpu().numpy()

    rdm = np.mean(rdm, axis=2)
    # remove diagonal
    rdm = rdm - np.diag(np.diag(rdm))

    return rdm




if __name__ == '__main__':
    test_df_path = r"C:\Users\shiri\Documents\School\Master\Research\Context\data\train_set_100_50.csv"
    results_dir = r"C:\Users\shiri\Documents\School\Master\Research\Context"

    test_df = pd.read_csv(test_df_path)


    current_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    #task = Task.init(project_name="face_recognition", task_name=f"test_{current_time}")
   # category_id_dict = get_category_id_dict_from_df(test_df)
    _, val_dataset = get_cat_datasets(test_df, augment_train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    #rdm = test(val_dataloader, model, category_id_dict)
