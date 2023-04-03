import torch.nn.functional as F
from ...helper import exp, maximum, zeros_like

"""
Loss functions
"""

def cross_entropy(y_hat, y):
    # using the torch implementation to account for extreme conditions
    return F.cross_entropy(y_hat, y)

def mse(y_hat, y):
    return (y_hat - y) ** 2 / 2
    # return F.mse_loss(y_hat, y)

"""
Activation functions
"""

def relu(x):
    return maximum(x, zeros_like(x))
    # return F.relu(x)

def sigmoid(x):
    return 1 / (1 + exp(-x))

def tanh(x):
    return (1 - exp(-2*x)) / (1 + exp(-2*x))
