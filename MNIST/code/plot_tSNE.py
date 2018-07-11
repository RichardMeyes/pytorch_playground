import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.colors import ListedColormap

import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.manifold import TSNE

def plot_single_digits(trainloader):
    for img in trainloader.dataset.train_data:
        npimg = img.numpy()
        plt.imshow(npimg, cmap='Greys')
        plt.show()


def plot_tSNE(trainloader, num_samples):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=5000, n_iter_without_progress=250,
                init='pca', random_state=1337)
    X_img = trainloader.dataset.train_data.numpy()[:num_samples]
    Y = trainloader.dataset.train_labels.numpy()[:num_samples]
    X = X_img.reshape(-1, X_img.shape[1]*X_img.shape[2])  # flattening out squared images for tSNE
    X_tsne = tsne.fit_transform(X)

    # scaling
    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    custom_cmap = plt.cm.Greys
    custom_cmap_colors = custom_cmap(np.arange(custom_cmap.N))
    custom_cmap_colors[:, -1] = np.linspace(0, 1, custom_cmap.N)
    custom_cmap = ListedColormap(custom_cmap_colors)

    if hasattr(offsetbox, 'AnnotationBbox'):
        for i_digit in range(num_samples):
            # correct color for plotting
            X_img[i_digit][X_img[i_digit, :, :] > 25] = 255
            X_img[i_digit][X_img[i_digit, :, :] <= 25] = 0
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(X_img[i_digit],
                                                                      cmap=custom_cmap,
                                                                      zoom=0.25),
                                                X_tsne[i_digit],
                                                frameon=False,
                                                pad=0)
            ax.add_artist(imagebox)
    ax.set_title("KL: {}".format(tsne.kl_divergence_))
    plt.savefig("../plots/MNIST_tSNE.png")
    # plt.show()


if __name__ == "__main__":

    # load data
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    # plot_single_digits(trainloader)
    print("plotting tSNE...")
    t0 = time.time()
    plot_tSNE(trainloader, num_samples=600)
    t1 = time.time()
    print("done! {0:.2f} seconds".format(t1 - t0))