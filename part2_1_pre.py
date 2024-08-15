import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
data = data.replace('-', 0).apply(pd.to_numeric)
t = data["t"].to_numpy()
x = data["x"].to_numpy()
y = data["y"].to_numpy()
plt.scatter(t, x, color='blue')
plt.legend()  
plt.show() 

plt.scatter(t, y, color='orange')
plt.legend()  
plt.show() 