import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torchvision
import torchvision.transforms as transforms


def plot_single_digits(trainloader):
    for img in trainloader.dataset.train_data:
        npimg = img.numpy()
        plt.imshow(npimg, cmap='Greys')
        plt.show()


def plot_tSNE(trainloader, num_samples):
    tsne = TSNE(n_components=2, perplexity=10, n_iter=10000, n_iter_without_progress=500,
                init='pca', random_state=1337)
    X = trainloader.dataset.train_data.numpy()[:num_samples]
    Y = trainloader.dataset.train_labels.numpy()[:num_samples]
    X = X.reshape(-1, X.shape[1]*X.shape[2])  # flattening out squared images for tSNE
    X_tsne = tsne.fit_transform(X)

    # scaling
    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    for i_digit in range(num_samples):
        ax.text(X_tsne[i_digit, 0], X_tsne[i_digit, 1], str(Y[i_digit]), color='r')
    ax.set_title("KL: {}".format(tsne.kl_divergence_))
    plt.show()


if __name__ == "__main__":

    # load data
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    # plot_single_digits(trainloader)
    plot_tSNE(trainloader, num_samples=600)