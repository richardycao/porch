# porch

Practice implementations of deep learning concepts. I'm stopping after implementing convolutional layers because training becomes too slow.

Install: `pip install --upgrade git+https://github.com/richardycao/porch.git#egg=porch`

Uninstall: `pip uninstall -y porch`

Custom implementations:
- porch.nn
    - (Layers)
        - Conv2d
        - Dropout
        - Flatten
        - Linear
        - MaxPool2d
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