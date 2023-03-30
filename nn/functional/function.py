import torch.nn.functional as F

def cross_entropy(y_hat, y):
    return F.cross_entropy(y_hat, y)

def mse(y_hat, y):
    return F.mse_loss(y_hat, y)

def relu(x):
    return F.relu(x)
