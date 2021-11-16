import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms


class DataFixture:
    pass


class ModelFixture:
    pass


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.max_pool2d = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(1)

        # explicitly define these layers for testing purposes
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        # x = F.max_pool2d(x, 2)
        # x = torch.flatten(x, 1)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class _MNISTFixture(DataFixture):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST("tmp/", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("tmp/", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    x, y = next(iter(train_loader))


class _Cifar10Fixture(DataFixture):
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = datasets.CIFAR10("tmp/", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10("tmp/", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    x, y = next(iter(train_loader))


class _ResNetFixture(ModelFixture):
    model = models.resnet18(pretrained=True)


Cifar10Fixture = _Cifar10Fixture()
ResNetFixture = _ResNetFixture()
