import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_data():
  data = torch.rand(1000, 2)
  label = ((data[:,0]+0.3*data[:,1]) > 0.5).to(torch.int)
  return data[:,0], label

input, label = generate_data()

# Make minibatches.
inputs = torch.split(input, 32)
labels = torch.split(label, 32)

# Define the two variables to optimize
b1 = torch.autograd.Variable(torch.tensor([0.01]), requires_grad=True)
b2 = torch.autograd.Variable(torch.tensor([0.01]), requires_grad=True)

#Define learning rate 
lr = 1
for epoch in range(15):
  batch_count = 0
  for x, y in zip(inputs,labels):
    # Calculate p_x as per formula above
    p_x = 1 / (torch.exp(-1*(b1 + b2 * x)) + 1)
    # Calculate the negative loss likelihood
    y = y.to(torch.float32)
    loss = torch.dot(y, torch.log(p_x))+torch.dot((1-y), torch.log(1-p_x))
    # Calculate the gradient of the loss w.r.t. the inputs
    gradient_loss_b1 = torch.mean(p_x - y)
    gradient_loss_b2 = torch.dot((p_x - y), x)/x.size(-1)
    # Update the parameters b according to SGD formula
    b1 = b1 - lr*gradient_loss_b1
    b2 = b2 - lr*gradient_loss_b2
    # Print out the loss value
    print(f'The Loss of Epoch {epoch+1}/15 Batch {batch_count+1} is {loss}')
    batch_count +=1
# Reproduce the image above to validate your result.
y_pred = 1 / (torch.exp(-1*(b1 + b2 * input)) + 1)
plt.figure(figsize=(8, 6)) 
plt.scatter(input.numpy(), y_pred.detach().numpy(), color='blue') 
plt.scatter(input.numpy(), label, color = 'red')
plt.legend()  
plt.show() 