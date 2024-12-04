import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 16
lr = 0.0008
window_size = 100

class Net(nn.Module):
    '''Model to regress 2d time series values given scalar input.'''
    def __init__(self, input_size=1, hidden_size_1=128, hidden_size_2=64, hidden_size_3=32, hidden_size_4=16, output_size=1):
        super(Net, self).__init__()
        # two shared fc layers
        self.window_size = window_size
        self.shared_layer1 = nn.Linear(input_size, hidden_size_1)
        self.shared_layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        
        self.shared_layer3 = nn.Linear(hidden_size_2, hidden_size_3)

        self.out_put_x = nn.Linear(hidden_size_3, output_size)
        # x is generated at this point
        self.shared_layer4 = nn.Linear(hidden_size_3, hidden_size_4)
        self.out_put_y = nn.Linear(hidden_size_4, output_size)

    def forward(self, x):
        x = x.reshape(-1,self.window_size, 1)
        x = F.relu(self.shared_layer1(x))
        x = F.relu(self.shared_layer2(x))        
        x = F.relu(self.shared_layer3(x))
        out_x = self.out_put_x(x)
        y = F.relu(self.shared_layer4(x))
        out_y = self.out_put_y(y)
        return torch.concat([out_x, out_y], dim =2)

class TimeSeriesDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        t = pd.to_numeric(data['t'])
        x = pd.to_numeric(data['x'].replace('-', np.nan))
        y = pd.to_numeric(data['y'].replace('-', np.nan))
        label = pd.concat([x, y], axis=1)
        self.label = torch.tensor(label.values, dtype= torch.float32)
        self.mask = ~torch.isnan(self.label)
        self.t = torch.tensor(t.values, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        return len(self.t) - self.window_size + 1

    def __getitem__(self, idx):
        feature = self.t[idx:idx+self.window_size]
        label = self.label[idx:idx+self.window_size, :]
        mask = self.mask[idx:idx+self.window_size, :]
        return feature, label, mask

class TimeSeriesDatasetTest(torch.utils.data.Dataset):

    def __init__(self):
        self.t = torch.linspace(start=0, end =100, steps=10001)
        self.window_size = window_size

    def __len__(self):
        return len(self.t) - self.window_size + 1

    def __getitem__(self, idx):
        feature = self.t[idx:idx+self.window_size]
        return feature


dataset = TimeSeriesDataset('data.csv')
trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
datasetTest = TimeSeriesDatasetTest()
testloader = torch.utils.data.DataLoader(datasetTest, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

def loss_fn(outputs, labels, mask):
  loss_x = criterion(outputs[:,:,0][mask[:,:,0]], labels[:,:,0][mask[:,:,0]])
  loss_y = criterion(outputs[:,:,1][mask[:,:,1]], labels[:,:,1][mask[:,:,1]])
  loss_x = loss_x if loss_x.numel() != 0 else torch.tensor(0,requires_grad=True, dtype=torch.float32)
  loss_y = loss_y if loss_y.numel() != 0 else torch.tensor(0,requires_grad=True, dtype=torch.float32)
  total_loss = 0.5*loss_x.mean()+ 0.5*loss_y.mean()
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
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fn(outputs, labels, mask)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 20 == 19:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
print('Finished Training')
torch.save(net.state_dict(), 'model_parameter_2_1_1')

def evaluation(trainloader):   
    t, x, y = [], [], []
    net.eval()
    with torch.no_grad():
        for i,data in enumerate(testloader):
            inputs = data
            pred = net(inputs)
            t.append(np.array(inputs.reshape(-1)))
            x.append(np.array(pred.reshape(-1, 2)[:,0]))
            y.append(np.array(pred.reshape(-1, 2)[:,1]))
    return np.concatenate(t), np.concatenate(x), np.concatenate(y)

t, x, y = evaluation(trainloader=trainloader)
data = pd.read_csv('data.csv')

t1 = pd.to_numeric(data['t'])
x1 = pd.to_numeric(data['x'].replace('-', np.nan))
y1 = pd.to_numeric(data['y'].replace('-', np.nan))

plt.figure()
plt.scatter(t1, x1, label="x", color='blue', s=0.5)
plt.scatter(t1, y1, label='y', color='orange', s=0.5)


plt.scatter(t, x, label="x", color='blue', s=0.5)
plt.scatter(t, y, label='y', color='orange', s=0.5)
plt.title('Regression Result')
plt.legend()
plt.savefig("figure/2_slide")
