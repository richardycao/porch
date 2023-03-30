import numpy as np

# all functions have incomplete parameters w.r.t. actual torch implementations

def matmul(x, y):
    return np.matmul(x, y)

def normal(mean, std, size):
    return np.random.normal(mean, std, size)

def rand(*args):
    return np.random.uniform(0, 1, size=args)

def sqrt(x):
    return np.sqrt(x)

