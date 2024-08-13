import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_data():
    data = torch.rand(1000, 2)
    label = ((data[:,0] + 0.3 * data[:,1]) > 0.5).to(torch.int)
    return data[:,0], label

input, label = generate_data()

# Make minibatches
inputs = torch.split(input, 32)
labels = torch.split(label, 32)

# Define the two variables to optimize
b1 = torch.tensor([0.01], requires_grad=True)
b2 = torch.tensor([0.01], requires_grad=True)

# Define learning rate 
lr = 0.2

for epoch in range(15):
    for x, y in zip(inputs, labels):
        # Reset gradients
        b1.grad = None
        b2.grad = None

        # Calculate p_x as per formula above
        z = b1 + b2 * x
        p_x = 1 / (1 + torch.exp(-z))

        # Calculate the negative log-likelihood loss
        y = y.to(torch.float32)  # Ensure y is float for log function
        loss = -torch.mean(y * torch.log(p_x) + (1 - y) * torch.log(1 - p_x))

        # Calculate the gradient of the loss w.r.t. the inputs
        loss.backward()

        # Update the parameters b according to SGD formula
        with torch.no_grad():
            b1 -= lr * b1.grad
            b2 -= lr * b2.grad

        # Print out the loss value
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Reproduce the image above to validate your result
y_pred = 1 / (1 + torch.exp(-(b1 + b2 * input)))
plt.figure(figsize=(8, 6))
plt.scatter(input.numpy(), label.numpy(), color='red', label='Data Points')
plt.plot(input.numpy(), y_pred.detach().numpy(), color='blue', label='Sigmoid Curve')
plt.xlabel('Input Feature')
plt.ylabel('Prediction Probability')
plt.title('Logistic Regression Prediction (Sigmoid Curve)')
plt.grid(True)
plt.legend()
plt.show()
