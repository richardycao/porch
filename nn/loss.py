import numpy as np

class Loss:
    def __init__(self):
        pass

    def __call__(self, y_hat, y):
        return self.loss(y_hat, y)
    
    def backward(self):
        raise NotImplementedError
 
    def item(self):
        raise NotImplementedError

    def loss(self, y_hat, y):
        raise NotImplementedError
    

class MSELoss(Loss):
    def __init__(self):
        pass

    def loss(self, y_hat, y):
        l = (y_hat - y) ** 2 /2
        return l.mean()

class CrossEntropyLoss(Loss):
    def __init__(self):
        pass