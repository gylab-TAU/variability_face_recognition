from sklearn.manifold import TSNE
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if __name__ == '__main__':
    cifar10_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # reshape the data
    cifar10_train_dataset.data = cifar10_train_dataset.data.reshape((len(cifar10_train_dataset.data), -1))
    #select only 1000 samples
    cifar10_train_dataset.data = cifar10_train_dataset.data[:4000]
    cifar10_train_dataset.targets = cifar10_train_dataset.targets[:4000]
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(cifar10_train_dataset.data)

    # Visualize the t-SNE results

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=cifar10_train_dataset.targets,
        palette=sns.color_palette("hls", 10),
        legend="full",
        alpha=0.5
    )
    plt.show()


