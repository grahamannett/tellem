import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

from dataclasses import dataclass

_USES_TORCH = True


@dataclass
class DataFixture:
    def __post_init__(self):
        if train_loader := getattr(self, "train_loader", None):
            self.x, self.y = next(iter(train_loader))


@dataclass
class ModelFixture:
    model: nn.Module = dataclasses.field(default=None, init=False)
    _default_out_features: int = None

    def __post_init__(self):
        pass

    def fix_num_classes(self, num_classes: int):
        last_layer = list(self.model.named_modules())[-1][1]
        self._default_out_features = last_layer.out_features
        last_layer.out_features = num_classes
        return self


class ConvNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.max_pool2d = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(1)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # explicitly define these layers for testing purposes
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
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
    num_classes: int = None

    # def fix_output(self, num_classes: int):

    #     # if self.num_classes is not None:
    #     num_ftrs = self.model.fc.in_features
    #     self.model.fc = nn.Linear(num_ftrs, num_classes)


# Data
Cifar10Fixture = _Cifar10Fixture()
MNISTFixture = _MNISTFixture()

# Models
ResNetFixture = ModelFixture(model=models.resnet18(pretrained=True))
BasicConvNet = ModelFixture(model=ConvNet())
