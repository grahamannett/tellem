import torch
import torch.nn.functional as F
from tellem.implementations import ImplementationBase

from tellem.testing import TestCase

_USES_TORCH = True


class TestImplementationsBase(TestCase):
    def setUp(self):
        super().setUp()

    def test_repr(self):
        impl = ImplementationBase()
