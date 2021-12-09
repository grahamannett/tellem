import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from torchvision.models import resnet18


import matplotlib.pyplot as plt
from tellem.implementations import GradCAM, CAM
from tellem.utils import upsample_to_image



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 5 * 5, 10)

    def forward(self, x: torch.Tensor):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        return x


# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose(
    [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# train_dataset = datasets.MNIST("tmp/data/", train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST("tmp/data/", train=False, transform=transform)

train_dataset = datasets.CIFAR10(root="tmp/data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="tmp/data", train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# model = Net()
model = resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)


# model.f

# model = resnet18(pretrained=True).eval()
x, y = next(iter(train_loader))

x = x[0]
x = x.unsqueeze(0)

y = y[0:1]
# y = F.one_hot(y[0], num_classes=10).unsqueeze(0)


cam = CAM(model)
grad_cam = GradCAM(model, loss_func=F.nll_loss)

cam.use_layer(conv_layer="layer4", fc_layer="fc")

grad_cam.use_layer(conv_layer="layer4")

cam_overlay = cam(x, y)

grad_cam_overlay = grad_cam(x, y)

overlay = upsample_to_image(x[0][0], cam_overlay)
plt.imshow(x[0][0])
plt.imshow(overlay, alpha=0.5)
plt.show()
