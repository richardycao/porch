from ...helper import empty, exp, log, max, maximum, squeeze, zeros_like

"""
Loss functions
"""

def cross_entropy(y_hat, y):
    z = empty(y.shape[0])
    a = max(y_hat, 1, keepdim=True).values
    logsumexp = squeeze(a) + log(exp(y_hat - a).sum(1))
    for i,k in enumerate(y):
        z[i] = y_hat[i,k] - logsumexp[i]
    return -z.mean()

def mse(y_hat, y):
    return (y_hat - y) ** 2 / 2

"""
Activation functions
"""

def relu(x):
    return maximum(x, zeros_like(x))

def sigmoid(x):
    return 1 / (1 + exp(-x))

def tanh(x):
    return (1 - exp(-2*x)) / (1 + exp(-2*x))
