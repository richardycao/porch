from .module import Module

class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        l = (y_hat - y) ** 2 /2
        return l.mean()

# class CrossEntropyLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, y_hat, y):
#         return 