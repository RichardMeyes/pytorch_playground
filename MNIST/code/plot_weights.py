import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 40, bias=False)
        self.fc2 = nn.Linear(40, 20, bias=False)
        self.fc3 = nn.Linear(20, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)  # needs NLLLos() loss
        return x

    def test_net(self):
        # test the net
        test_loss = 0
        correct = 0
        for data, target in testloader:
            data, target = Variable(data), Variable(target)
            data = data.view(-1, 28 * 28)
            net_out = self(data)
            # sum up batch loss
            test_loss += criterion(net_out, target).data.item()
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).sum().item()
        test_loss /= len(testloader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct,
                                                                                     len(testloader.dataset),
                                                                                     100. * correct / len(
                                                                                         testloader.dataset)))
        acc = 100.0 * correct / len(testloader.dataset)
        return acc


def plot_weights(weights, scale, label):
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)
    for i in range(40):
        ax = fig.add_subplot(6, 7, i+1)
        i_weights = weights[i].reshape(28, 28)
        ax.matshow(i_weights, cmap='seismic', vmin=scale[0]-np.mean(scale), vmax=scale[1]-np.mean(scale))
        ax.set_xticks(())
        ax.set_yticks(())
    plt.suptitle("{0}, accuracy: {1}%".format(label, acc))
    plt.savefig("../plots/weights_" + label)
    plt.clf()
    # plt.show()


if __name__ == "__main__":

    # load net
    net = Net()
    criterion = nn.NLLLoss()  # nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # load fully trained network
    net.load_state_dict(torch.load('../nets/MNIST_MLP(40, 20, 10).pt'))
    net.eval()

    # load data and test network
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

    # plot fully trained network weights
    # weights = (net.fc1.weight.data.numpy().T + net.fc1.bias.data.numpy()).T  # biases considered
    weights = net.fc1.weight.data.numpy()
    scale = (np.min(weights), np.max(weights))
    acc = net.test_net()
    plot_weights(weights, scale, label="full")

    # modify net, test accuracy and plot weights
    for i_unit in range(40):
        net.load_state_dict(torch.load('../nets/MNIST_MLP(40, 20, 10).pt'))
        net.eval()
        net.fc1.weight.data[i_unit, :] = torch.zeros(784)
        # weights = (net.fc1.weight.data.numpy().T + net.fc1.bias.data.numpy()).T  # biases considered
        weights = net.fc1.weight.data.numpy()
        acc = net.test_net()
        plot_weights(weights, scale, label="knockout_" + str(i_unit+1))

