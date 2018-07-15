import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.colors import ListedColormap

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from train_test_net import Net


def plot_single_digits(trainloader):
    for img in trainloader.dataset.train_data:
        npimg = img.numpy()
        plt.imshow(npimg, cmap='Greys')
        plt.show()


def plot_tSNE(testloader, labels, num_samples, name=None, title=None):
    X_img = testloader.dataset.test_data.numpy()[:num_samples]

    print("loading fitted tSNE coordinates...")
    X_tsne = pickle.load(open("../data/tSNE/X_tSNE_{0}.p".format(num_samples), "rb"))

    print("plotting tSNE...")
    # scaling
    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)
    t0 = time.time()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # Define custom color maps
    custom_cmap_black = plt.cm.Greys
    custom_cmap_black_colors = custom_cmap_black(np.arange(custom_cmap_black.N))
    custom_cmap_black_colors[:, -1] = np.linspace(0, 1, custom_cmap_black.N)
    custom_cmap_black = ListedColormap(custom_cmap_black_colors)

    custom_cmap_red = plt.cm.bwr
    custom_cmap_red_colors = custom_cmap_red(np.arange(custom_cmap_red.N))
    custom_cmap_red_colors[:, -1] = np.linspace(0, 1, custom_cmap_red.N)
    custom_cmap_red = ListedColormap(custom_cmap_red_colors)

    custom_cmap_white = plt.cm.Greys
    custom_cmap_white_colors = custom_cmap_white(np.arange(custom_cmap_white.N))
    custom_cmap_white_colors[:, -1] = 0
    custom_cmap_white = ListedColormap(custom_cmap_white_colors)

    color_maps = [custom_cmap_red, custom_cmap_black, custom_cmap_white]

    if hasattr(offsetbox, 'AnnotationBbox'):
        for i_digit in range(num_samples):
            # correct color for plotting
            X_img[i_digit][X_img[i_digit, :, :] > 10] = 255
            X_img[i_digit][X_img[i_digit, :, :] <= 10] = 0
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(X_img[i_digit],
                                                                      cmap=color_maps[labels[i_digit]],
                                                                      zoom=0.25),
                                                X_tsne[i_digit],
                                                frameon=False,
                                                pad=0)
            ax.add_artist(imagebox)

    ax.set_title(title)
    # save figure
    plt.savefig("../plots/MNIST_tSNE_{0}_{1}.png".format(num_samples, name), dpi=1200)
    t1 = time.time()
    print("done! {0:.2f} seconds".format(t1 - t0))


if __name__ == "__main__":
    # setting rng seed for reproducability
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    num_digits = testloader.dataset.test_labels.size()

    net_trained = Net()
    net_trained.to(device)
    criterion = nn.NLLLoss()  # nn.CrossEntropyLoss()
    net_trained.load_state_dict(torch.load('../nets/MNIST_MLP(20, 10)_trained.pt'))
    net_trained.eval()
    full_acc, labels = net_trained.test_net(criterion, testloader, device)

    # plot_single_digits(trainloader)
    labels[labels == 0] = -1
    plot_tSNE(testloader, labels, num_samples=10000, title="accuray: {0}%".format(full_acc))

    # plot tSNE for knocked out units
    for i_unit in range(20):
        net_trained.load_state_dict(torch.load('../nets/MNIST_MLP(20, 10)_trained.pt'))
        net_trained.eval()
        net_trained.fc1.weight.data[i_unit, :] = torch.zeros(784)
        acc, labels_ko = net_trained.test_net(criterion, testloader, device)

        labels_ko[labels == -1] = -1
        plot_tSNE(testloader, labels_ko, num_samples=10000, name="ko_" + str(i_unit+1),
                  title="knockout_" + str(i_unit + 1) + ", accuray: {0}%, delta_acc: {1:.2f}%".format(acc,
                                                                                                      full_acc - acc))