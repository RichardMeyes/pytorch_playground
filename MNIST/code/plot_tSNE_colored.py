import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.colors import ListedColormap

import torch
import torchvision
import torchvision.transforms as transforms


def plot_single_digits(trainloader):
    for img in trainloader.dataset.train_data:
        npimg = img.numpy()
        plt.imshow(npimg, cmap='Greys')
        plt.show()


def plot_tSNE(testloader, num_samples):
    X_img = testloader.dataset.test_data.numpy()[:num_samples]
    Y = testloader.dataset.test_labels.numpy()[:num_samples]

    print("loading fitted tSNE coordinates...")
    X_tsne = pickle.load(open("../data/tSNE/X_tSNE_{0}.p".format(num_samples), "rb"))

    print("plotting tSNE...")
    t0 = time.time()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # define class colors
    cmaps = [plt.cm.bwr,
             plt.cm.bwr,
             plt.cm.Wistia,
             plt.cm.Greys,
             plt.cm.cool,
             plt.cm.Purples,
             plt.cm.coolwarm,
             plt.cm.bwr,
             plt.cm.PiYG,
             plt.cm.cool]

    if hasattr(offsetbox, 'AnnotationBbox'):
        for i_digit in range(num_samples):
            # create colormap
            custom_cmap = cmaps[Y[i_digit]]
            custom_cmap_colors = custom_cmap(np.arange(custom_cmap.N))
            if Y[i_digit] in [7, 6, 9]:
                custom_cmap_colors = custom_cmap_colors[::-1]
            custom_cmap_colors[:, -1] = np.linspace(0, 1, custom_cmap.N)
            custom_cmap = ListedColormap(custom_cmap_colors)

            # correct color for plotting
            X_img[i_digit][X_img[i_digit, :, :] > 10] = 255
            X_img[i_digit][X_img[i_digit, :, :] <= 10] = 0
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(X_img[i_digit],
                                                                      cmap=custom_cmap,
                                                                      zoom=0.25),
                                                X_tsne[i_digit],
                                                frameon=False,
                                                pad=0)
            ax.add_artist(imagebox)
    plt.savefig("../plots/MNIST_tSNE_{0}_colored.png".format(num_samples), dpi=1200)
    # plt.show()
    t1 = time.time()
    print("done! {0:.2f} seconds".format(t1 - t0))


if __name__ == "__main__":

    # load data
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    num_digits = testloader.dataset.test_labels.size()

    # plot_single_digits(trainloader)
    plot_tSNE(testloader, num_samples=10000)