from tellem.implementations.base import ImplementationBase
from tellem.types import Tensor, Model


import torch

_USES_TORCH = True


class SimpleTaylorDecomposition(ImplementationBase):
    """https://arxiv.org/abs/1706.07979

    Args:
        ImplementationBase ([type]): [description]
    """

    def __init__(self, model: Model = None, **kwargs):
        super().__init__(model=model, **kwargs)

    def __call__(self, x: Tensor, y: Tensor = None):
        x.requires_grad = True

        x.register_hook()
        preds = self.model(x)

        # get gradients
        grad = torch.grad(x, preds)


class LayerwiseRelevanceDecomposition(ImplementationBase):
    """https://arxiv.org/abs/1706.07979
    https://git.tu-berlin.de/gmontavon/lrp-tutorial
    """

    def __init__(self, model: Model = None, **kwargs):
        super().__init__(model=model, **kwargs)

    def _lrp(self, layer, activations, R):
        # section 6.3 of """https://arxiv.org/abs/1706.07979
        # seems like its done for tf, need to redo for pytorch
        clone = layer.clone()
        clone.W = max(0, layer.W)
        clone.B = 0
        z = clone.forward(activations)
        s = R / z
        c = clone.backward(s)
        return activations * c


class SimpleLayerwiseRelevenceDecomposition(ImplementationBase):

    def __init__(self, model: Model = None, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)

    def __check_model__(self):
        pass

    