import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from train_test_net import Net


def plot_weights(weights, scale, unit_struct, title, name):
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.2, hspace=0.4)
    for i in range(20):
        ax = fig.add_subplot(4, 5, i+1)
        i_weights = weights[i].reshape(28, 28)
        ax.matshow(i_weights, cmap='seismic', vmin=scale[0]-np.mean(scale), vmax=scale[1]-np.mean(scale))
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title("T: {0:.2e}, p: {1:.2e}".format(int(unit_struct[i, 0]), unit_struct[i, 1]))
    plt.suptitle("{0}".format(title))
    plt.savefig("../plots/weights_" + name)
    plt.close()
    # plt.show()


def calc_unit_struct_metric(weights_trained, weights_untrained):
    unit_struct = np.zeros((len(weights_trained), 2))
    for i_unit in range(len(weights_trained)):
        s, p = spst.wilcoxon(weights_trained[i_unit], weights_untrained[i_unit])
        unit_struct[i_unit] = s, p
    return unit_struct


if __name__ == "__main__":

    device = torch.device("cpu:0")

    # load nets and weights
    net_trained = Net()
    net_untrained = Net()
    criterion = nn.NLLLoss()  # nn.CrossEntropyLoss()
    net_trained.load_state_dict(torch.load('../nets/MNIST_MLP(20, 10)_trained.pt'))
    net_trained.eval()
    net_untrained.load_state_dict(torch.load('../nets/MNIST_MLP(20, 10)_untrained.pt'))
    net_untrained.eval()

    unit_struct_untrained = calc_unit_struct_metric(net_untrained.fc1.weight.data.numpy()+0.001, net_untrained.fc1.weight.data.numpy())
    unit_struct = calc_unit_struct_metric(net_trained.fc1.weight.data.numpy(), net_untrained.fc1.weight.data.numpy())

    # load data and test network
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

    # plot fully trained network weights
    # weights = (net.fc1.weight.data.numpy().T + net.fc1.bias.data.numpy()).T  # biases considered
    weights = net_trained.fc1.weight.data.numpy()
    scale = (np.min(weights), np.max(weights))
    acc_full = net_trained.test_net(criterion, testloader, device)
    plot_weights(weights, scale, unit_struct, title="trained accuracy: {0}%".format(acc_full), name="full")

    # plot untrained network weights
    weights = net_untrained.fc1.weight.data.numpy()
    acc_untrained = net_untrained.test_net(criterion, testloader, device)
    plot_weights(weights, scale, unit_struct_untrained,
                 title="untrained accuracy: {0}%".format(acc_untrained), name="0full")

    # modify net, test accuracy and plot weights
    for i_unit in range(20):
        net_trained.load_state_dict(torch.load('../nets/MNIST_MLP(20, 10)_trained.pt'))
        net_trained.eval()
        net_trained.fc1.weight.data[i_unit, :] = torch.zeros(784)
        # weights = (net.fc1.weight.data.numpy().T + net.fc1.bias.data.numpy()).T  # biases considered
        weights = net_trained.fc1.weight.data.numpy()
        acc = net_trained.test_net(criterion, testloader, device)
        plot_weights(weights, scale, unit_struct,
                     title="knockout_" + str(i_unit+1) + ", accuray: {0}%, delta_acc: {1:.2f}%".format(acc, acc_full-acc),
                     name="knockout_" + str(i_unit+1))

