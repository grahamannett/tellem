import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms

from tellem.implementations import GradCAM, CAM

from tests.helpers import TestCase
from torchvision.models import resnet18


class TestCAM(TestCase):
    def setUp(self):
        super().setUp()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST("tmp/", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST("tmp/", train=False, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        self.model = models.resnet18(pretrained=True)

        # self.model = resnet18(pretrained=True).eval()
        self.x, self.y = next(iter(self.train_loader))

    def test_cam(self):
        x = self.x[0]
        x = x.unsqueeze(0)
        y = self.y[0]

        cam = CAM(self.model)
        # cam.use_linear_layer("fc")
        cam.use_layer(conv_layer="layer4", fc_layer="fc")
        # cam_overlay = cam(x, y)

    