import numpy as np
from ..utils import rand, sqrt, matmul

class Parameter

class Module:
    def __init__(self, training=True):
        self.training = training
        self.parameters = []

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        raise NotImplementedError
    
    def parameters(self):
        return self.parameters

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        sqrt_k = sqrt(1 / in_features)
        self.weight = rand(out_features, in_features) * 2 * sqrt_k - sqrt_k
        self.bias = rand(out_features) * 2 * sqrt_k - sqrt_k if bias else 0
    
    def forward(self, x):
        return matmul(x, self.weight.T) + self.bias
    