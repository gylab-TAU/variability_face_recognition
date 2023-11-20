import os
import numpy as np
from utils import get_TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import random

np.random.seed(42)
random.seed(42)

DATASET_NAMES = [['low_var_train', 'low_var_sim', 'low_var_dis'], ['high_var_train', 'high_var_sim', 'high_var_dis']]


def plot_histogram_of_distance_from_mean_all_classes(data_dir, low_sim, low_dis, high_sim, high_dis, num_classes,  num_per_set):
    total_distance_mat = np.zeros((num_classes, 4, num_per_set))
    for class_num, em_path in enumerate(os.listdir(data_dir)):
        X = np.load(os.path.join(data_dir, em_path))
        X_scaled = scaler.fit_transform(X)

        u_embeddings = np.mean(X_scaled, axis=0)
        dist_from_mean = np.sqrt(np.sum((X_scaled - u_embeddings) ** 2, axis=1))

        sorted_idx = np.argsort(dist_from_mean)
        # add u_embeddings to X
        X_scaled = np.vstack((X_scaled, u_embeddings))

        distance_mat = np.zeros((4, num_per_set))
        lims = [(0, low_sim), (low_sim, low_dis), (0, high_sim), (high_sim, high_dis)]
        for i in range(len(lims)):
            bottom_lim = lims[i][0]
            top_lim = lims[i][1]
            relevant_idx = sorted_idx[bottom_lim:top_lim].copy()
            chosen_idx = random.sample(list(relevant_idx), num_per_set)
            chosen_dist = dist_from_mean[chosen_idx]
            distance_mat[i, :] = chosen_dist

        total_distance_mat[class_num, :, :] = distance_mat

    datasets_names = ['low_var_sim', 'low_var_dis', 'high_var_sim', 'high_var_dis']

    for i in range(len(datasets_names)):
        dataset_name = datasets_names[i]
        dataset_idx_mat = total_distance_mat[:, i, :]
        plt.hist(dataset_idx_mat.flatten(), bins=5, alpha=0.5, label=dataset_name)
    plt.legend()
    plt.title('Distance from mean, 1 classes')
    plt.ylabel('Number of samples')
    plt.xlabel('Distance from mean')
    plt.show()

def plot_histogram_of_distance_from_mean_all_classes_dict(data_dir, low_sim, low_dis, high_sim, high_dis, num_classes,  train_set_size, test_set_size):
    total_train_distance_mat = np.zeros((num_classes, 2, train_set_size))
    total_test_distance_mat = np.zeros((num_classes, 4, test_set_size))
    for class_num, em_path in enumerate(os.listdir(data_dir)):
        X = np.load(os.path.join(data_dir, em_path))
        X_scaled = scaler.fit_transform(X)

        u_embeddings = np.mean(X_scaled, axis=0)
        dist_from_mean = np.sqrt(np.sum((X_scaled - u_embeddings) ** 2, axis=1))

        sorted_idx = np.argsort(dist_from_mean)
        # add u_embeddings to X
        X_scaled = np.vstack((X_scaled, u_embeddings))

        train_distances_mat = np.zeros((2, train_set_size))
        test_distances_mat = np.zeros((4, test_set_size))
        lims = [(0, low_sim, 'train_test'), (0, high_sim,'train_test'), (low_sim, low_dis,'test'),  (high_sim, high_dis,'test')]
        for i in range(len(lims)):
            bottom_lim = lims[i][0]
            top_lim = lims[i][1]
            relevant_idx = sorted_idx[bottom_lim:top_lim].copy()
            if lims[i][2]=='train_test':
                chosen_train_idx = random.sample(list(relevant_idx), train_set_size)
                chosen_train_dist = dist_from_mean[chosen_train_idx]
                train_distances_mat[i, :] = chosen_train_dist
                chosen_test_idx = np.setdiff1d(relevant_idx, chosen_train_idx)
                chosen_test_idx = random.sample(list(chosen_test_idx), test_set_size)
                chosen_test_dist = dist_from_mean[chosen_test_idx]
                test_distances_mat[i, :] = chosen_test_dist
            else:
                chosen_idx = random.sample(list(relevant_idx), test_set_size)
                chosen_dist = dist_from_mean[chosen_idx]
                test_distances_mat[i, :] = chosen_dist


        total_train_distance_mat[class_num, :, :] = train_distances_mat
        total_test_distance_mat[class_num, :, :] = test_distances_mat

    train_datasets_names = ['low_var_train', 'high_var_train']
    test_datasets_names = ['low_var_sim',  'high_var_sim','low_var_dis', 'high_var_dis']

    for i in range(len(train_datasets_names)):
        dataset_name = train_datasets_names[i]
        dataset_idx_mat = total_train_distance_mat[:, i, :]
        plt.hist(dataset_idx_mat.flatten(), bins=10, alpha=0.5, label=dataset_name)
    plt.legend()
    plt.title('Train - Distance from mean, All classes')
    plt.ylabel('Number of samples')
    plt.xlabel('Distance from mean')
    plt.xlim(12,30)
    plt.show()

    for i in range(len(test_datasets_names)):
        dataset_name = test_datasets_names[i]
        dataset_idx_mat = total_test_distance_mat[:, i, :]
        plt.hist(dataset_idx_mat.flatten(), bins=10, alpha=0.5, label=dataset_name)
    plt.legend()
    plt.title('Test - Distance from mean, All classes')
    plt.ylabel('Number of samples')
    plt.xlabel('Distance from mean')
    plt.xlim(12,30)
    plt.show()

def plot_detailed_histogram_of_distance_from_mean_one_class(data_dir, low_sim, low_dis, high_sim, high_dis, train_set_size, test_set_size):
    em_path = os.listdir(data_dir)[2]

    X = np.load(os.path.join(data_dir, em_path))
    X_scaled = scaler.fit_transform(X)

    u_embeddings = np.mean(X_scaled, axis=0)
    dist_from_mean = np.sqrt(np.sum((X_scaled - u_embeddings) ** 2, axis=1))

    sorted_idx = np.argsort(dist_from_mean)

    low_var_indices = sorted_idx[:low_sim]
    low_var_train_indices = np.random.choice(low_var_indices, train_set_size, replace=False)
    low_var_test_similar_indices = np.setdiff1d(low_var_indices, low_var_train_indices)
    low_var_test_similar_indices = np.random.choice(low_var_test_similar_indices, test_set_size, replace=False)
    low_var_test_dissimilar_indices = np.random.choice(
        sorted_idx[low_sim:low_dis], test_set_size, replace=False)

    high_var_indices = sorted_idx[0:high_sim].copy()
    high_var_train_indices = np.random.choice(high_var_indices, train_set_size, replace=False)
    high_var_test_similar_indices = np.setdiff1d(high_var_indices, high_var_train_indices)#[0:test_set_size]
    tmp = high_var_test_similar_indices.copy()
    high_var_test_similar_indices = np.random.choice(tmp, test_set_size, replace=False)
    tmp2 = sorted_idx[high_sim:high_dis].copy()
    high_var_test_dissimilar_indices = np.random.choice(
        tmp2, test_set_size, replace=False)



    datasets_names = [['low_var_train','low_var_sim', 'low_var_dis'],['high_var_train', 'high_var_sim', 'high_var_dis']]
    datasets = [[low_var_train_indices, low_var_test_similar_indices, low_var_test_dissimilar_indices], [high_var_train_indices, high_var_test_similar_indices, high_var_test_dissimilar_indices]]
    # datasets_names = [['low_var_train'],
    #                   ['high_var_train']]
    # datasets = [[low_var_train_indices],
    #             [high_var_train_indices]]

    plt.subplot(3, 1, 1)
    plt.hist(dist_from_mean,bins=20)
    plt.xlim(12, 35)
    plt.ylim(0, 35)
    for dataset_type, names in zip(datasets, datasets_names):
        plt.subplot(3, 1, datasets_names.index(names)+2)
        for dataset,name in zip(dataset_type,names):
             distances = dist_from_mean[dataset]
             plt.hist(distances, bins=5, alpha=0.5, label=name)
        plt.xlim(12,35)
        plt.ylim(0, 35)
        plt.legend()
    # plt.title('Distance from mean, 1 classes')
    # plt.ylabel('Number of samples')
    # plt.xlabel('Distance from mean')
    plt.show()


def plot_tsne_distance_from_mean_all_classes( data_dir):
    for class_num, em_path in enumerate(os.listdir(data_dir)):
        X = np.load(os.path.join(data_dir, em_path))
        X_scaled = scaler.fit_transform(X)

        u_embeddings = np.mean(X_scaled, axis=0)
        dist_from_mean = np.sqrt(np.sum((X_scaled - u_embeddings) ** 2, axis=1))

        sorted_idx = np.argsort(dist_from_mean)
        # add u_embeddings to X
        X_scaled = np.vstack((X_scaled, u_embeddings))

        idx_list = []
        for i in range(0, len(sorted_idx), 100):
            idx_list.append(sorted_idx[i:i + 100])
        idx_list.append(len(X_scaled)-1)


        tsne_result = get_TSNE(X_scaled)
        color_list = ['red',  'blue', 'black']
        for idx, color in zip(idx_list, color_list):
            plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1], c=color, s=50)
        plt.show()


if __name__ == '__main__':
    DATA_DIR = r'C:\Users\shiri\Documents\School\Master\high_low_var_datasets\distance_by_batches\june24\embeddings_resnet'
    MIN_INSTANCES_NUM = 200
    LOW_SIM = np.int(MIN_INSTANCES_NUM*0.25)
    LOW_DIS = np.int(MIN_INSTANCES_NUM*0.55)
    HIGH_SIM = np.int(MIN_INSTANCES_NUM*0.65)
    HIGH_DIS = np.int(MIN_INSTANCES_NUM*0.9)
    TRAIN_SIZE = 35
    TEST_SIZE = 15
    NUM_CLASSES = len(os.listdir(DATA_DIR))

    #plot_histogram_of_distance_from_mean_all_classes(DATA_DIR, LOW_SIM, LOW_DIS, HIGH_SIM, HIGH_DIS, NUM_CLASSES, TRAIN_SIZE)

    plot_histogram_of_distance_from_mean_all_classes_dict(DATA_DIR, LOW_SIM, LOW_DIS, HIGH_SIM, HIGH_DIS, NUM_CLASSES, TRAIN_SIZE, TEST_SIZE)


   # plot_tsne_distance_from_mean_all_classes(DATA_DIR )

    #plot_detailed_histogram_of_distance_from_mean_one_class(DATA_DIR, LOW_SIM, LOW_DIS, HIGH_SIM, HIGH_DIS, TRAIN_SIZE, TEST_SIZE)







