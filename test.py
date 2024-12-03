import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('data.csv')
t = pd.to_numeric(data['t'], errors='coerce')
x = pd.to_numeric(data['x'], errors='coerce')
y = pd.to_numeric(data['y'], errors='coerce')
label = pd.concat([x, y], axis=1)
label = torch.tensor(label.values, dtype= torch.float32)
mask = ~torch.isnan(label)
print(label)
print(mask)