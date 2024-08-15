import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 10
window_size = 50

class Net(nn.Module):
    '''Model to regress 2d time series values given scalar input.'''
    def __init__(self, num_neuron_layer1, num_neuron_layer2, window_size):
        super(Net, self).__init__()

        # input
        self.input_layer = nn.Linear()

    def forward(self, x):
        #TODO

class TimeSeriesDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, window_size):
        data = pd.read_csv(csv_file)
        dash_row = data.applymap(lambda x: x == '-').any(axis = 1)
        self.data = dash_row[~dash_row].pply(pd.to_numeric)
        self.window_size = window_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t = torch.tensor(self.data.iloc[idx:idx + self.window_size].values, dtype=torch.float64)
        processed_data = torch.tensor(self.data.iloc[idx:idx + self.window_size].values, dtype=torch.float64)
        return processed_data[0], processed_data[1:]

dataset = TimeSeriesDataset('data.csv')
trainloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

def loss_fn(outputs, labels):
  #TODO
optimizer = #TODO what is a good optimizer?

net = Net()

for epoch in range(300):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training')