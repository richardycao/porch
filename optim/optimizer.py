import numpy as np

class Optimizer:
    def __init__(self):
        pass

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, params, lr, momentum=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        
        self.params = params
        self.lr = lr
        self.momentum = momentum

        # setting these to 0 for now
        self.weight_decay = 0
        self.dampening = 0
        self.nesterov = False
        self.maximize = False

    def step(self):
        pass

    