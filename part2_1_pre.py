import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# data = pd.read_csv('data.csv')
# data = data.replace('-', 0).apply(pd.to_numeric)
# t = data["t"].to_numpy()
# x = data["x"].to_numpy()
# y = data["y"].to_numpy()
# plt.scatter(t, x, color='blue')
# plt.legend()  
# plt.show() 

# plt.scatter(t, y, color='orange')
# plt.legend()  
# plt.show() 

data = pd.read_csv('data.csv')
x = pd.to_numeric(data['x'].replace('-', np.nan)).interpolate(method='spline', order=3).fillna(method='ffill').fillna(method='bfill')
y = pd.to_numeric(data['y'].replace('-', np.nan)).interpolate(method='spline', order=3).fillna(method='ffill').fillna(method='bfill')
label = pd.concat([x, y], axis =1)
label = torch.tensor(label.values, dtype= torch.float32)
print(label)