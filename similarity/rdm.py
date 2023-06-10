import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from numpy.linalg import norm
from PIL import Image
import csv
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
from models import get_vgg_pretrained_vggface2, get_vgg_pretrained_imagenet, get_resnet_model
from sim_transforms import normalize, Rescale

mtcnn = MTCNN(image_size=224, post_process=False)
cos = torch.nn.CosineSimilarity()

composed_transforms = transforms.Compose([transforms.ToTensor(), Rescale((224, 224)), normalize])


def load_data(data_dir, num_classes=None, instance_num_per_class=None):
    """
    get list of paths to images in data_dir
    :param data_dir: path to data directory
    :param num_classes: num classes to load. If None, load all classes
    :param instance_num_per_class: num instances per class to load. If None, load all instances
    :return: list of paths to images
    """
    paths_list = []
    names = []
    classes_list = os.listdir(data_dir)
    if num_classes:
        classes_list = classes_list[:num_classes]
    for c in classes_list:
        class_path = os.path.join(data_dir, c)
        imgs = os.listdir(class_path)
        if instance_num_per_class:
            imgs = imgs[:instance_num_per_class]
        for img in imgs:
            im_path = os.path.join(class_path, img)
            paths_list.append(im_path)
            names.append(f'{c}_{img}')

    return paths_list, names


def get_model(model_type, model_weights=None):
    if model_type == 'vgg_vggface2':
        if model_weights is None:
            raise TypeError('model weights must be provided for vgg_vggface2')
        model = get_vgg_pretrained_vggface2(model_weights)
        embedding_size = 4096
    elif model_type == 'vgg_imagenet':
        model = get_vgg_pretrained_imagenet()
        embedding_size = 4096
    elif model_type == 'resnet_vggface2':
        model = get_resnet_model(pretrain='vggface2')
        embedding_size = 512
    elif model_type == 'resnet_casia':
        model = get_resnet_model(pretrain='casia-webface')
        embedding_size = 512
    else:
        raise ValueError('model type not supported')
    return model, embedding_size


def get_embeddings(data, model_type, perform_mtcnn=True):
    model, embedding_size = get_model('vgg_vggface2', args.model_path)
    embeddings = np.zeros((len(data), embedding_size))

    for i, im_path in enumerate(data):
        img = Image.open(im_path)
        if img.mode != 'RGB':  # PNG imgs are RGBA
            img = img.convert('RGB')
        if perform_mtcnn:
            img = mtcnn(img)
            # if mtcnn post_processing = true, image is standardised to -1 to 1, and size 160x160.
            # else, image is not standardised ([0-255]), and size is 160x160
            img = img / 255.0
            img = normalize(img)
        else:  # in case images are already post mtcnn. In this case need to rescale to 160x160 and normalize
            img = composed_transforms(img)

        if model_type == ('vgg_vggface2' or 'vgg_imagenet'):
            img_embedding = model(img.unsqueeze(0).float())[0]['fc7']
        else:
            img_embedding = model(img.unsqueeze(0).float())

        embeddings[i] = img_embedding.detach().numpy()

    return embeddings


def get_rdm(embeddings):
    rdm = np.zeros((len(embeddings), len(embeddings)))
    for i, first_em in enumerate(embeddings):
        for j, second_em in enumerate(embeddings):
            rdm[i, j] = np.dot(first_em, second_em) / (norm(first_em) * norm(second_em))
    print(rdm)
    return rdm


def save_rdm_matrix_with_names(rdm, names, save_path):
    n = len(names)
    assert rdm.shape == (n, n)

    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row with the names
        writer.writerow([''] + names)

        # Write the distances matrix with names
        for i in range(n):
            row = [names[i]] + rdm[i].tolist()
            writer.writerow(row)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-data_dir", "--data_dir", dest="data_dir", help="folder with data")
    parser.add_argument("-model_path", "--model_path", dest="model_path", help="path to model weights", required=False)
    parser.add_argument("-results_path", "--results_path", dest="results_path", help="path to save csv result",
                        required=False)
    args = parser.parse_args()

    data_paths_list, names_list = load_data(args.data_dir, num_classes=10, instance_num_per_class=10)

    model_embeddings = get_embeddings(data_paths_list, 'vgg_vggface2', perform_mtcnn=True)
    rdm_matrix = get_rdm(model_embeddings)

    if not args.results_path:
        print(f'results path not provided, saving to data_dir {args.data_dir}/rdm.csv ')
        args.results_path = os.path.join(args.data_dir, 'rdm.csv')

    save_rdm_matrix_with_names(rdm_matrix, names_list, args.results_path)

    plt.imshow(rdm_matrix)
    plt.show()
