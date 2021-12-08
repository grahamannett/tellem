import torch.nn.functional as F

from tellem.attacks.fgsm import FastGradientSignMethod
from tellem.testing import Cifar10Fixture, ResNetFixture, TestCase


class TestGradCAM(TestCase):
    def setup_method(self, *method):
        ResNetFixture.fix_output(num_classes=10)
        self.use_fixture(ResNetFixture)
        self.use_fixture(Cifar10Fixture)

        self.x_ = self.x
        self.y_ = self.y

        self.x = self.x[0].unsqueeze(0)
        self.y = self.y[0]

        self.loss_func = F.nll_loss

    def test_fgsm(self):
        fgsm = FastGradientSignMethod(self.model, self.loss_func, eps=0.01)

        attack_x = fgsm(self.x, self.y)
        assert fgsm
