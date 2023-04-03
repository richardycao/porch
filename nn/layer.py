from torch import nn
from .module import Module
import math
from ..helper import empty, flatten, matmul, rand

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()

        self.weight = nn.Parameter(empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(empty(out_features))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        return matmul(x, self.weight.T) + self.bias
    
class Dropout(Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        mask = (rand(x.shape) > self.p).float()
        return (mask * x) / (1 - self.p)
    
class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return flatten(x, start_dim=self.start_dim, end_dim=self.end_dim)
    