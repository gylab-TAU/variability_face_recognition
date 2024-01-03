import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader

from contrastive.model import SiameseNetwork
from dataset import TripletDataset
from config import TrainConfig as config
contrastive_model_output_size = 128

path = r"C:\Users\shiri\Documents\School\Master\Research\Context\weights\contrastive_model_no_context_100_epochs.pth"
model = SiameseNetwork(embedding_dim=contrastive_model_output_size)
model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
model.eval()


train_dataset = TripletDataset(config.data_dir, config.num_classes, config.num_samples_per_class,
                                   config.split_to_categories, config.num_categories)
print (train_dataset.categories_to_classes)
# split the dataset in train and test set
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) * 0.8),
                                                                            len(train_dataset) - int(
                                                                                len(train_dataset) * 0.8)])
#split val_dataset to val and test
val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [int(len(val_dataset) * 0.8),
                                                                            len(val_dataset) - int(
                                                                                len(val_dataset) * 0.8)])


test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)


with torch.no_grad():
    features = []
    labels = []
    for batch_idx, (anchor, _, _, label) in enumerate(test_dataloader):
        inputs, label = anchor, label
        features.append(model(inputs).cpu().numpy())
        labels.append(label.cpu().numpy())

features = np.concatenate(features)
labels = np.concatenate(labels)

clustering = AgglomerativeClustering(n_clusters=10).fit(features)
plt.figure(figsize=(16, 10))
plt.plot(clustering.labels_, labels, 'o')
plt.xlabel('clustering')
plt.ylabel('labels')
plt.title('clustering vs labels')
plt.show()
