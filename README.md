# porch

Practice implementations of deep learning concepts.

Install: `pip install --upgrade git+https://github.com/richardycao/porch.git#egg=porch`

Uninstall: `pip uninstall -y porch`

Custom implementations:
- porch.nn
    - (Layers)
        - BatchNorm1d
        - BatchNorm2d
        - Conv2d
        - Dropout
        - Flatten
        - Linear
        - MaxPool2d
        - Sequential
        - ReLU
    - (Losses)
        - MSE
        - CrossEntropyLoss
    - Functional
        - cross_entropy
        - mse
        - relu
        - sigmoid
        - tanh
- porch.optim
    - SGD

Everything else has been copied from torch.