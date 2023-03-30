from .module import Module
from .functional.function import cross_entropy, mse

class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        return mse(y_hat, y).mean()

class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        return cross_entropy(y_hat, y).mean()