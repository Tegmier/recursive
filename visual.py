import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 5
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
        x = self.layer_4(x)
        return x

class TimeSeriesDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        t = pd.to_numeric(data['t'])
        x = pd.to_numeric(data['x'].replace('-', np.nan))
        y = pd.to_numeric(data['y'].replace('-', np.nan))
        label = pd.concat([x, y], axis=1)
        self.label = torch.tensor(label.values, dtype= torch.float32)
        self.mask = ~torch.isnan(self.label)
        # print(self.label)
        # print(self.mask)
        self.t = torch.tensor(t.values, dtype=torch.float32)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        feature = self.t[idx]
        label = self.label[idx, :]
        mask = self.mask[idx, :]
        return feature, label, mask
    

dataset = TimeSeriesDataset('data.csv')
trainloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

def loss_fn(outputs, labels, mask):

  loss_x = criterion(outputs[:,0][mask[:,0]], labels[:,0][mask[:,0]])
  loss_y = criterion(outputs[:,1][mask[:,1]], labels[:,1][mask[:,1]])
#   loss_x = criterion(outputs[:,0], labels[:,0])[mask[:,0]]
#   loss_y = criterion(outputs[:,1], labels[:,1])[mask[:,1]]
  loss_x = loss_x if loss_x.numel() != 0 else torch.tensor(0,requires_grad=True, dtype=torch.float32)
  loss_y = loss_y if loss_y.numel() != 0 else torch.tensor(0,requires_grad=True, dtype=torch.float32)
#   print(loss_x, loss_y)
  total_loss = 0.5*loss_x.mean()+ 0.5*loss_y.mean()
#   print(total_loss)
  return total_loss

criterion = nn.MSELoss(reduction='none')
net = Net()
net.load_state_dict(torch.load('model_parameter'))

def evaluation(trainloader):
    t, x, y = [], [], []
    net.eval()
    with torch.no_grad():
        for i,data in enumerate(trainloader, 0):
            inputs, labels, mask = data
            pred = net(inputs)
            t.append(np.array(inputs.reshape(-1)))
            x.append(np.array(pred.reshape(-1, 2)[:,0]))
            y.append(np.array(pred.reshape(-1, 2)[:,1]))
    return np.concatenate(t), np.concatenate(x), np.concatenate(y)

t, x, y = evaluation(trainloader=trainloader)

plt.figure()
# plt.scatter(t, x, label="x", color='blue', s=1)
# plt.scatter(t, y, label='y', color='orange', s=1)
# plt.title('Regression Result')
# plt.legend()

sorted_indices = np.argsort(t)
t_sorted = t[sorted_indices]
x_sorted = x[sorted_indices]
y_sorted = y[sorted_indices]
plt.plot(t_sorted, x_sorted, label='x', color='blue')
plt.plot(t_sorted, y_sorted, label='y', color='orange')
plt.show()
