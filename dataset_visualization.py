import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

t = pd.to_numeric(data['t'])
x = pd.to_numeric(data['x'].replace('-', np.nan))
y = pd.to_numeric(data['y'].replace('-', np.nan))

plt.figure()
plt.scatter(t, x, label="x", color='blue', s=1)
plt.scatter(t, y, label='y', color='orange', s=1)
plt.show()