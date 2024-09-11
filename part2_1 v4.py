import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 4
lr = 0.001


class Net(nn.Module):
    '''Model to regress 2d time series values given scalar input.'''
    def __init__(self, input_size=1, output_size=1, hidden_size=256):
        super(Net, self).__init__()

        # input
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer_1 = nn.Linear(hidden_size, output_size)
        self.output_layer_2 = nn.Linear(hidden_size, output_size)
        self.MLP_layers = 6
        self.batch_norm = nn.BatchNorm1d(num_features=hidden_size)
        self.mlp1 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.MLP_layers)])
        self.mlp2 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.MLP_layers)])
        self.mlp3 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.MLP_layers)])


    def forward(self, x):

        x = x.reshape(BATCH_SIZE, -1)
        x = self.input_layer(x)
        for layer in self.mlp1:
            x = self.batch_norm(F.relu(layer(x)))

        for layer in self.mlp2:
            out1 = self.batch_norm(F.relu(layer(x)))

        # 通过第三个 MLP
        for layer in self.mlp3:
            out2 = self.batch_norm(F.relu(layer(out1)))

        out1 = self.output_layer_1(out1)
        out2 = self.output_layer_2(out2)

        return out1, out2

class TimeSeriesDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        t = pd.to_numeric(data['t'])
        x = pd.to_numeric(data['x'].replace('-', np.nan))
        y = pd.to_numeric(data['y'].replace('-', np.nan))

        self.label_x = torch.tensor(x.values, dtype=torch.float32).cuda()
        self.label_y = torch.tensor(y.values, dtype=torch.float32).cuda()

        self.mask_x = ~torch.isnan(self.label_x).cuda()
        self.mask_y = ~torch.isnan(self.label_y).cuda()
        self.t = torch.tensor(t.values, dtype=torch.float32).cuda()

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        feature = self.t[idx]
        label_x = self.label_x[idx]
        label_y = self.label_y[idx]
        mask_x = self.mask_x[idx]
        mask_y = self.mask_y[idx]
        return feature, label_x, label_y, mask_x, mask_y
    

dataset = TimeSeriesDataset('data.csv')
trainloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

def loss_fn(output_x, output_y, label_x, label_y, mask_x, mask_y):

  loss_x = criterion(output_x[mask_x], label_x[mask_x])
  loss_y = criterion(output_y[mask_y], label_y[mask_y])
  loss_x = loss_x/loss_x.numel() if loss_x.numel() != 0 else torch.tensor(0,requires_grad=True, dtype=torch.float32).cuda()
  loss_y = loss_y/loss_y.numel() if loss_y.numel() != 0 else torch.tensor(0,requires_grad=True, dtype=torch.float32).cuda()
  total_loss = 0.5*loss_x.mean()+ 0.5*loss_y.mean()
  return total_loss

criterion = nn.MSELoss(reduction='none').cuda()
# criterion = nn.SmoothL1Loss(reduction='none')
net = Net().cuda()
# optimizer = optim.Adam(net.parameters(), lr=lr)
optimizer = optim.AdamW(net.parameters(), lr=lr)
# optimizer = optim.SGD(net.parameters(), lr=lr)

for epoch in range(50):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, label_x, label_y, mask_x, mask_y = data
        optimizer.zero_grad()
        output_x, output_y = net(inputs)
        loss = loss_fn(output_x, output_y, label_x, label_y, mask_x, mask_y)
        loss.backward()
        running_loss += loss.item()
        if i % 20 == 19:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
print('Finished Training')
torch.save(net.state_dict(), 'model_parameter_v3')

def evaluation(trainloader):
    t, x, y = [], [], []
    net.eval()
    with torch.no_grad():
        for i,data in enumerate(trainloader, 0):
            inputs, label_x, label_y, mask_x, mask_y = data
            output_x, output_y = net(inputs)
            t.append(np.array(inputs.reshape(-1).cpu()))
            x.append(np.array(output_x.cpu()))
            y.append(np.array(output_y.cpu()))
    return np.concatenate(t), np.concatenate(x), np.concatenate(y)

t, x, y = evaluation(trainloader=trainloader)

plt.figure()
plt.scatter(t, x, label="x", color='blue', s=1)
plt.scatter(t, y, label='y', color='orange', s=1)
plt.title('Regression Result')
plt.legend()
plt.show()
