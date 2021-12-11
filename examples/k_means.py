from sklearn.datasets import load_iris
import torch
import torch.nn as nn


iris = load_iris()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.pool = nn.MaxPool1d(1)

    def forward(self, x):
        x = self.linear(x)
        # x = -self.pool(-x)
        breakpoint()
        # x = self.pool(x)
        return x


net = Net()

X = iris.data  # we only take the first two features.
y = iris.target

x = torch.from_numpy(X).to(torch.float32)

preds = net(x[0])
breakpoint()
