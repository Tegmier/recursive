import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 4
lr = 0.0008


class Net(nn.Module):
    '''Model to regress 2d time series values given scalar input.'''
    def __init__(self, input_size=1, output_size=1, hidden_size_1=256, hidden_size_2=128, hidden_size_3=64, hidden_size_4=16):
        super(Net, self).__init__()

        # input
        self.layer_1 = nn.Linear(input_size, hidden_size_1)
        self.layer_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer_3_1 = nn.Linear(hidden_size_2, hidden_size_3)
        self.layer_3_2 = nn.Linear(hidden_size_2, hidden_size_3)
        self.layer_4_1 = nn.Linear(hidden_size_3, hidden_size_4)
        self.layer_4_2 = nn.Linear(hidden_size_3, hidden_size_4)
        self.layer_5_1 = nn.Linear(hidden_size_4, output_size)
        self.layer_5_2 = nn.Linear(hidden_size_4, output_size)

    def forward(self, x):
        x = x.reshape(BATCH_SIZE, -1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x_3 = F.relu(self.layer_3_1(x))
        y_3 = F.relu(self.layer_3_2(x))
        x_4 = F.relu(self.layer_4_1(x_3))
        y_4 = F.relu(self.layer_4_2(y_3))
        x_5 = F.relu(self.layer_5_1(x_4))
        y_5 = F.relu(self.layer_5_2(y_4))
        return torch.concat([x_5, y_5], dim=1)

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

# criterion = nn.MSELoss(reduction='none')
criterion = nn.SmoothL1Loss(reduction='none')
net = Net()
# optimizer = optim.Adam(net.parameters(), lr=lr)
optimizer = optim.AdamW(net.parameters(), lr=lr)
# optimizer = optim.SGD(net.parameters(), lr=lr)

for epoch in range(300):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels, mask = data
        # print(f'input.shape: {inputs.shape}')
        # print(f'labels.shape: {labels.shape}')
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_fn(outputs, labels, mask)
        loss.backward()
        # with torch.no_grad():
        #     for param in net.parameters():
        #         if param.grad is not None:
        #             valid_grad = param.grad
        #             if valid_grad == torch.nan:  # 检查是否存在有效梯度
        #                 param.grad.zero_()  # 如果没有有效梯度，则清零
        optimizer.step()
        # print statistics
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
            inputs, labels, mask = data
            pred = net(inputs)
            t.append(np.array(inputs.reshape(-1)))
            x.append(np.array(pred.reshape(-1, 2)[:,0]))
            y.append(np.array(pred.reshape(-1, 2)[:,1]))
    return np.concatenate(t), np.concatenate(x), np.concatenate(y)

t, x, y = evaluation(trainloader=trainloader)

plt.figure()
plt.scatter(t, x, label="x", color='blue', s=1)
plt.scatter(t, y, label='y', color='orange', s=1)
plt.title('Regression Result')
plt.legend()
plt.show()
