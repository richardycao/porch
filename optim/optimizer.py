from torch.optim import Optimizer
from ..helper import zeros_like

class SGD(Optimizer):
    def __init__(self, params, lr, momentum=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        self.momentum = momentum
        self.state = dict()
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(mom=zeros_like(p.data))

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p not in self.state:
                    self.state[p] = dict(mom=zeros_like(p.data))
                mom = self.state[p]['mom']
                mom = self.momentum * mom - group['lr'] * p.grad.data
                p.data += mom
    