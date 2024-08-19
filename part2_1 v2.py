import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 10
lr = 0.0005


class Net(nn.Module):
    '''Model to regress 2d time series values given scalar input.'''
    def __init__(self, input_size=1, output_size=2, hidden_size_1=64, hidden_size_2=32, hidden_size_3=16):
        super(Net, self).__init__()

        # input
        self.layer_1 = nn.Linear(input_size, hidden_size_1)
        self.layer_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer_3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.layer_4 = nn.Linear(hidden_size_3, output_size)

    def forward(self, x):
        x = x.reshape(BATCH_SIZE, -1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        return self.layer_4(x)

class TimeSeriesDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        t = pd.to_numeric(data['t'])
        x = pd.to_numeric(data['x'].replace('-', np.nan)).interpolate(method='spline', order=3).fillna(method='ffill').fillna(method='bfill')
        y = pd.to_numeric(data['y'].replace('-', np.nan)).interpolate(method='spline', order=3).fillna(method='ffill').fillna(method='bfill')
        label = pd.concat([x, y], axis =1)
        self.label = torch.tensor(label.values, dtype= torch.float32)
        self.t = torch.tensor(t.values, dtype=torch.float32)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        feature = self.t[idx]
        label = self.label[idx, :]
        return feature, label
    

dataset = TimeSeriesDataset('data.csv')
trainloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

def loss_fn(outputs, labels):
  return criterion(outputs, labels)

criterion = nn.MSELoss()
net = Net()
# optimizer = optim.Adam(net.parameters(), lr=lr)
optimizer = optim.AdamW(net.parameters(), lr=lr)
# optimizer = optim.SGD(net.parameters(), lr=lr)


for epoch in range(300):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # print(f'input.shape: {inputs.shape}')
        # print(f'labels.shape: {labels.shape}')
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

def evaluation(trainloader):
    t, x, y = [], [], []
    net.eval()
    with torch.no_grad():
        for i,data in enumerate(trainloader, 0):
            inputs, labels = data
            pred = net(inputs)
            t.append(np.array(inputs.reshape(-1)))
            x.append(np.array(pred.reshape(-1, 2)[:,0]))
            y.append(np.array(pred.reshape(-1, 2)[:,1]))

    return np.concatenate(t), np.concatenate(x), np.concatenate(y)

t, x, y = evaluation(trainloader=trainloader)

plt.figure(figsize=(8, 6))
plt.scatter(t, x, label="x", color='blue')
plt.scatter(t, y, label='y', color='orange')
plt.title('Regression Result')
plt.legend()
plt.show()
