import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

print(len(data))

# data = data[(data['x']!='-') & (data['y']!='-')]
print(len(data))

t = pd.to_numeric(data['t'])
x = pd.to_numeric(data['x'].replace('-', np.nan))
y = pd.to_numeric(data['y'].replace('-', np.nan))

print(t)


x_filled = np.copy(x)
x_filled[np.isnan(x)] = 0
fft_coeffs = np.fft.rfft(x_filled)
freqs = np.fft.rfftfreq(len(t), d=(t[1] - t[0]))
fft_coeffs[np.abs(freqs) > 1] = 0
x_reconstructed = np.fft.irfft(fft_coeffs, n=len(t))


plt.figure()
plt.scatter(t, x, label="x", color='blue', s=1)
plt.scatter(t, x_reconstructed, label="x", color='blue', s=0.1)
# plt.scatter(t, y, label='y', color='orange', s=1)
plt.savefig("figure/dataset_visulation.png")