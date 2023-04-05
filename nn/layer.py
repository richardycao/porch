from torch import nn
from .module import Module
import math
from ..helper import empty, flatten, matmul, mul, rand, unsqueeze
    
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple) and len(kernel_size) == 2:
            self.kernel_size = kernel_size
        else:
            raise Exception("kernel_size must be int or tuple of length 2")
        
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple) and len(stride) == 2:
            self.stride = stride
        else:
            raise Exception("stride must be int or tuple of length 2")
        
        if isinstance(padding, int):
            self.padding = (padding, padding)
        elif isinstance(padding, tuple) and len(padding) == 2:
            self.padding = padding
        else:
            raise Exception("padding must be int or tuple of length 2")

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.K = nn.Parameter(empty(out_channels, in_channels, *self.kernel_size))
        nn.init.normal_(self.K)
        if bias:
            self.bias = nn.Parameter(empty(1))
        else:
            self.register_parameter('bias', None)


    def forward(self, x):
        # Apply padding
        b, c, w, h = x.shape
        pad = empty((b, c, w+self.padding[0]*2, h+self.padding[1]*2))
        pad[:,:,self.padding[0]:self.padding[0]+w, self.padding[1]:self.padding[1]+h] = x
        x = pad

        # Convolutions
        y = empty((b, 
                   self.out_channels, 
                   int((w - self.K.shape[2] + self.padding[0]*2 + self.stride[0]) / self.stride[0]), 
                   int((h - self.K.shape[3] + self.padding[1]*2 + self.stride[1]) / self.stride[1])))
        for i in range(y.shape[2]):
            for j in range(y.shape[3]):
                corr2d = mul(self.K, unsqueeze(x[:,:,i:i+self.kernel_size[0],j:j+self.kernel_size[1]], 1).repeat(1,self.out_channels, 1, 1, 1))
                y[:,:,i,j] = flatten(corr2d, start_dim=2).sum(2)
        return y + self.bias if self.bias != None else y

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
    
