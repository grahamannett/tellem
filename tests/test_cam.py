import torch.nn.functional as F

from tellem.implementations import CAM, GradCAM
from tellem.testing import Cifar10Fixture, ResNetFixture, TestCase


class TestCAM(TestCase):
    def setUp(self):
        super().setUp()
        self.use_fixture(ResNetFixture)
        self.use_fixture(Cifar10Fixture)

        self.x = self.x[0].unsqueeze(0)
        self.y = self.y[0]
        self.args = TestCase.kwargs_helper(base=self, conv_layer="layer4", fc_layer="fc")

    def test_cam(self):
        cam = CAM(self.model)
        cam.use_layer(conv_layer="layer4", fc_layer="fc")
        results = cam.get_cam(self.x, self.y)

        assert results.ndim == 2

    def test_auto(self):
        results = CAM.__tellem_function__(self.args)
        assert results.ndim == 2


class TestGradCAM(TestCase):
    def setup_method(self, *method):
        self.use_fixture(ResNetFixture)
        self.use_fixture(Cifar10Fixture)
        self.x = self.x[0].unsqueeze(0)
        self.y = self.y[0]
        self.loss_func = F.nll_loss

    def test_gradcam(self):
        gradcam = GradCAM(self.model, self.loss_func)
        gradcam.use_layer("layer4")
        results = gradcam.get_gradcam(self.x, self.y)
        assert results.ndim == 2

    def test_auto(self):
        self.args = TestCase.kwargs_helper(base=self, conv_layer="layer4")
        results = GradCAM.__tellem_function__(self.args)
        assert results.ndim == 2
