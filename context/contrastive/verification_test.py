import os
import numpy as np
import pandas as pd
import random
import torch
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from numpy.linalg import norm

from contrastive.model import SiameseNetwork
from faces.utils import Rescale


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)

transform = transforms.Compose([transforms.ToTensor(),Rescale((64, 64)),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

def create_test_set(data_dir, num_classes):
    df_test = pd.DataFrame(columns=['image_path','image_class', 'pair_path','pair_class', 'label'])
    # data dir has folder with different classes. i want to create a dataframe with positive paris - pairs of
    # images from the same class, and negative pairs - pairs of images from different classes.
    # for each class, i want to create 10 positive pairs and 10 negative pairs.
    # i want to create 40 classes, so i need 400 positive pairs and 400 negative pairs.

    # get all classes
    dataset = {}
    classes = os.listdir(data_dir)
    classes = [c for c in classes if os.path.isdir(os.path.join(data_dir, c))]
    # get all images in each class
    for c in classes[0:num_classes]:
        images = os.listdir(os.path.join(data_dir, c))
        images = [i for i in images if os.path.isfile(os.path.join(data_dir, c, i))]
        try:
            dataset[c] = random.choices(images, k=30)
        except:
            print (f"{c} has less than 30 images")
    # create positive pairs
    for c in dataset.keys():
        for i in range(10):
            pair = random.choices(dataset[c], k=2)
            df_test = df_test.append({'image_path': os.path.join(data_dir, c, pair[0]),
                                      'image_class': c,
                                      'pair_path': os.path.join(data_dir, c, pair[1]),
                                        'pair_class': c,
                                      'label': 1}, ignore_index=True)
    # create negative pairs
    for c in dataset.keys():
        for i in range(10):
            class_path = os.path.join(data_dir, c)
            image = random.choice(dataset[c])
            other_classes = [other for other in dataset.keys() if other != c]
            other_class = random.choice(other_classes)
            other_class_path = os.path.join(data_dir, other_class)
            df_test = df_test.append({'image_path': os.path.join(class_path, image),
                                        'image_class': c,
                                        'pair_path': os.path.join(other_class_path, random.choice(dataset[other_class])),
                                        'pair_class': other_class,
                                        'label': 0}, ignore_index=True)
    df_test.to_csv(os.path.join(r"C:\Users\shiri\Documents\School\Master\Research\Context\data\test.csv"), index=False)
    return df_test

def inference_same_different(test_df, model_path, device):
    model = SiameseNetwork(embedding_dim=128)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    results_df = pd.DataFrame(columns=['similarity', 'label', 'image_path', 'pair_path'])
    features = []
    classes = []
    for i, row in test_df.iterrows():
        image = row['image_path']
        pair = row['pair_path']
        label = row['label']
        image = Image.open(image)
        pair = Image.open(pair)
        image = transform(image)
        pair = transform(pair)
        image, pair = image.to(device), pair.to(device)
        image_output = model.forward_one(image.float().unsqueeze(0))
        pair_output = model.forward_one(pair.float().unsqueeze(0))
        features.append(image_output.cpu().detach().numpy())
        features.append(pair_output.cpu().detach().numpy())
        classes.append(row['image_class'])
        classes.append(row['pair_class'])
        sim = torch.cosine_similarity(image_output, pair_output)
        results_df = results_df.append({'similarity': sim.item(), 'label': label,
                                        'image_path': row['image_path'], 'pair_path': row['pair_path']},
                                         ignore_index=True)


    return results_df, features, classes


def calc_roc(results_df):
    y_true = results_df['label'].values
    y_true = [np.float64(y) for y in y_true]
    y_scores = results_df['similarity'].values
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    plt.plot(fpr, tpr)
    plt.title(f"ROC curve, AUC = {roc_auc}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    plt.close()
    return roc_auc


def plot_tsne(features, labels):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(np.concatenate(features))
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=labels,
        palette=sns.color_palette("hls", len(set(labels))),
    )
    plt.show()
    plt.close()


def get_rdm(embeddings):
    embeddings = np.concatenate(embeddings)
    rdm = np.zeros((len(embeddings), len(embeddings)))
    for i, first_em in enumerate(embeddings):
        for j, second_em in enumerate(embeddings):
            rdm[i, j] = np.dot(first_em, second_em) / (norm(first_em) * norm(second_em))
    return rdm


if __name__ == '__main__':
    data_dir = r"C:\Users\shiri\Documents\School\Master\Research\Face_recognition\Data\VGG-Face2\MTCNN\train_mtcnn_aligned_2000"
    num_classes = 40
    create_new_test_set = True

    if create_new_test_set:
       df_test = create_test_set(data_dir, num_classes)
    else:
        df_test = pd.read_csv(os.path.join(r"C:\Users\shiri\Documents\School\Master\Research\Context\data\test.csv"))

    model_path = r"C:\Users\shiri\Documents\School\Master\Research\Context\weights\contrastive_model_40_classes_0.3_context.pth"
   # model_path = r"C:\Users\shiri\PycharmProjects\variability_face_recognition\context\contrastive\faces\contrastive_model.pth"
    results_df, features, classes = inference_same_different(df_test, model_path, device)
    results_df.to_csv(r"C:\Users\shiri\Documents\School\Master\Research\Context\data\results.csv", index=False)
    plot_tsne(features, classes)
    roc_auc = calc_roc(results_df)
    rdm = get_rdm(features)
    plt.imshow(rdm)
    plt.show()

