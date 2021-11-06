import torch
import torch.nn as nn
import torchvision
import typer

from dataclasses import dataclass

from tellem.implementations import TCAV


@dataclass
class DataHelper:
    train: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader


class CNN(torch.nn.Module):
    """Basic CNN architecture."""

    def __init__(self, in_channels=1, n_classes: int = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 8, 1)  # (batch_size, 3, 28, 28) --> (batch_size, 64, 21, 21)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, 6, 2)  # (batch_size, 64, 21, 21) --> (batch_size, 128, 8, 8)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 128, 5, 1)  # (batch_size, 128, 8, 8) --> (batch_size, 128, 4, 4)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(128 * 4 * 4, 128)  # (batch_size, 128, 4, 4) --> (batch_size, 2048)
        self.fc2 = nn.Linear(128, n_classes)  # (batch_size, 128) --> (batch_size, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def load_data():
    train_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root="data/", download=True, transform=train_transforms)
    test_dataset = torchvision.datasets.MNIST(root="data/", train=False, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    return DataHelper(train=train_loader, test=test_loader)


def main():
    data = load_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = CNN().to(device)

    tcav = TCAV.TCAV(net)
    tcav.use_layers()
    # for epoch in range(0, 5):
    #     net.train()
    #     train_loss = 0.0

    #     for x, y in data.train:
    #         x, y = x.to(device), y.to(device)

    # # define concepts/noise

    # concepts = data.test.data[data.test.y == 1]
    # non_concept = noise_like(concepts)


if __name__ == "__main__":
    typer.run(main)