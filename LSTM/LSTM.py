import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out[-1].view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out


if __name__ == "__main__":

    t = np.linspace(0, 10, 1000, endpoint=True)
    data = np.sin(2*np.pi*t)
    # plt.plot(t, data)
    # plt.show()


    def sliding_windows(data, seq_length):
        x = []
        y = []

        for i in range(len(data) - seq_length - 1):
            _x = data[i:(i + seq_length)]
            _y = data[i + seq_length]
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y)


    seq_length = 7
    learning_rate = 0.01
    input_size = 1
    hidden_size = 5
    num_layers = 3
    num_epochs = 50

    x, y = sliding_windows(data, seq_length)

    train_size = int(len(y) * 0.7)
    test_size = len(y) - train_size
    trainX = torch.Tensor(np.array(x[0:train_size]))
    trainX = Variable(trainX).view(trainX.size(0), trainX.size(1), -1)
    testX = torch.Tensor(np.array(x[train_size:len(x)]))
    testX = Variable(testX).view(testX.size(0), testX.size(1), -1)
    trainY = torch.Tensor(np.array(y[0:train_size]))
    trainY = Variable(trainY).view(trainY.size(0), -1)
    testY = torch.Tensor(np.array(y[train_size:len(y)]))
    testY = Variable(testY).view(testY.size(0), -1)



    lstm = LSTM(input_size, hidden_size, num_layers)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        outputs = lstm(trainX)
        optimizer.zero_grad()
        # obtain the loss function
        loss = criterion(outputs, trainY)
        loss.backward()
        optimizer.step()
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.data[0]))

    lstm.eval()
    test_predict = lstm(testX)

    test_predict = test_predict.data.numpy()
    testY = testY.data.numpy()
    plt.plot(testY)
    plt.plot(test_predict)
    plt.show()
