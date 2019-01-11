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

    def imshow(img):
        img = img / 2 + 0.5  # un-normalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # plot some data
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 6, bias=False)
        self.fc2 = nn.Linear(6, 4, bias=False)
        self.fc3 = nn.Linear(4, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)  # needs NLLLos() loss
        return x

    def train_net(self, criterion, optimizer, trainloader, epochs, device):
        # save untrained net
        if save:
            torch.save(net.state_dict(), '../nets/MNIST_MLP(20, 10)_untrained.pt')
        # train the net
        log_interval = 10
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = Variable(data), Variable(target)
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
                                                                                   len(trainloader.sampler),
                                                                                   100. * batch_idx / len(trainloader),
                                                                                   loss.data.item()))
        if save:
            # save trained net
            torch.save(net.state_dict(), '../nets/MNIST_MLP(20, 10)_trained.pt')

    def test_net(self, criterion, testloader, device):
        # test the net
        test_loss = 0
        correct = 0
        correct_class = np.zeros(10)
        correct_labels = np.array([], dtype=int)
        for i_batch, (data, target) in enumerate(testloader):
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28 * 28)
            net_out = self(data)
            # sum up batch loss
            test_loss += criterion(net_out, target).data.item()
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
            batch_labels = pred.eq(target.data)
            correct_labels = np.append(correct_labels, batch_labels)
            for i_label in range(len(target)):
                label = target[i_label].item()
                correct_class[label] += batch_labels[i_label].item()
            correct += batch_labels.sum()
        test_loss /= len(testloader.sampler)  # must consider the length of the sampler rather than of the whole dataset in case of subsampling!
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct,
                                                                                     len(testloader.sampler),
                                                                                     100. * correct.item() / len(
                                                                                         testloader.sampler)))
        acc = 100. * correct.item() / len(testloader.sampler)
        # calculate class_acc
        acc_class = np.zeros(10)
        for i_label in range(10):
            num = (testloader.dataset.test_labels.numpy() == i_label).sum()
            acc_class[i_label] = correct_class[i_label]/num
        return acc, correct_labels, acc_class


if __name__ == "__main__":

    # setting random seed
    np.random.seed(2478)
    torch.manual_seed(1273890)
    torch.cuda.manual_seed(23789)
    # torch.cuda.seed()

    """ setting flags """
    # chose data plotting
    plot = False
    # chose CPU or GPU:
    dev = "GPU"
    # chose training or loading pre-trained model
    train = True
    save = False
    test = True
    # chose to train with a specific subset of digits. Right now, only pairs are allowed
    use_subset = True
    subset_digits = [0, 1]  # specifiy which pair of digits should be used for training and testing

    # prepare GPU
    if dev == "GPU":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu:0")
    print("current device:", device)

    # build net
    net = Net()
    if dev == "GPU":
        print("sending net to GPU")
        net.to(device)
    criterion = nn.NLLLoss()  # nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # load data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    # Train with a
    if use_subset:
        mask0 = trainset.train_labels.numpy() == subset_digits[0]
        mask1 = trainset.train_labels.numpy() == subset_digits[1]
        mask_train = mask0 | mask1
        mask0 = testset.test_labels.numpy() == subset_digits[0]
        mask1 = testset.test_labels.numpy() == subset_digits[1]
        mask_test = mask0 | mask1
        my_sampler_train = torch.utils.data.SubsetRandomSampler(np.squeeze(np.argwhere(mask_train)))
        my_sampler_test = torch.utils.data.SubsetRandomSampler(np.squeeze(np.argwhere(mask_test)))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, sampler=my_sampler_train,
                                                        shuffle=False, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, sampler=my_sampler_test,
                                                        shuffle=False, num_workers=4)

    if plot:
        plot_data()

    if train:
        net.train_net(criterion, optimizer, trainloader, epochs=5, device=device)
    else:
        net.load_state_dict(torch.load('../nets/MNIST_MLP(20, 10)_trained.pt'))
        net.eval()

    if test:
        acc, correct_labels, acc_class = net.test_net(criterion, testloader, device)
        print(acc)
        print(correct_labels)
        print(acc_class)
        print(acc_class.mean())  # NOTE: This does not equal to the calculated total accuracy as the distribution of labels is not equal in the test set!