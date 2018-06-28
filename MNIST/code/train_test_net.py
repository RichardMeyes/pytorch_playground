import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable


def plot_data():
    # plot some data
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_net(self, epochs):
        # train the net
        log_interval = 10
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = Variable(data), Variable(target)
                if dev == "GPU":
                    data, target = data.to(device), target.to(device)
                # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
                data = data.view(-1, 28 * 28)
                optimizer.zero_grad()
                net_out = self(data)
                loss = criterion(net_out, target)
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                                   len(trainloader.dataset),
                                                                                   100. * batch_idx / len(trainloader),
                                                                                   loss.data.item()))
        if save:
            # save trained net
            torch.save(net.state_dict(), '../nets/MNIST_MLP(20, 20, 10).pt')

    def test_net(self):
        # test the net
        test_loss = 0
        correct = 0
        for data, target in testloader:
            data, target = Variable(data), Variable(target)
            if dev == "GPU":
                data, target = data.to(device), target.to(device)
            data = data.view(-1, 28 * 28)
            net_out = self(data)
            # sum up batch loss
            test_loss += criterion(net_out, target).data.item()
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).sum()
        test_loss /= len(testloader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                     len(testloader.dataset),
                                                                                     100. * correct / len(
                                                                                         testloader.dataset)))


def imshow(img):
    img = img / 2 + 0.5  # un-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":

    """ setting flags """
    # chose data plotting
    plot = False
    # chose CPU or GPU:
    dev = "GPU"
    # chose training or loading pre-trained model
    train = True
    save = True
    test = True

    # prepare GPU
    if dev == "GPU":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu:0")
    print("current device:", device)

    # build net
    net = Net()
    if dev == "GPU":
        net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # load data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    if plot:
        plot_data()

    if train:
        net.train_net(epochs=100)
    else:
        net.load_state_dict(torch.load('../nets/MNIST_MLP(20, 20, 10).pt'))
        net.eval()

    if test:
        net.test_net()