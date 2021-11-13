from typing import Any, Callable, List

from tellem import Capture
from tellem.implementations.base import ImplementationBase
from tellem.types import ConvLayer, LinearLayer, Model, Tensor


class CAM(ImplementationBase):
    """

    Official Paper:
        - https://arxiv.org/pdf/1512.04150.pdf

    Official Implementation:
    Unofficial Implementations:
        - https://github.com/frgfm/torch-cam

    Args:
        ImplementationBase ([type]): [description]


    Useage:
        model = Model()
    """

    def __init__(self, model: Model = None, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.capture = {}
        self.check_model()

    def get_cam(self, x: Tensor, y: Tensor, **kwargs):
        _ = self.model(x)

        # x should be one sample so remove dim on activations
        activations = self.activations.squeeze(0)

        # get the weights for the class
        weights = self.weights[y, :].squeeze(0)

        # other library does:  weights[(...,) + (None,) * missing_dims]
        while weights.ndim < activations.ndim:
            weights = weights.unsqueeze(-1)

        cam_vals = weights * activations

        # then sum over feature maps
        if cam_vals.ndim > 2:
            cam_vals = cam_vals.sum(dim=0)

        return cam_vals

    def check_model(self):
        # conv feature maps → global average pooling → softmax layer
        modules = list(self.model.named_modules())
        last_linear_layer = None
        last_conv_layer = None
        for idx, (_, layer_obj) in enumerate(modules):
            if isinstance(layer_obj, ConvLayer):
                last_conv_layer = idx
            if isinstance(layer_obj, LinearLayer):
                last_linear_layer = idx

        if last_conv_layer < last_linear_layer:
            return

        raise TypeError("Linear Layer Must Follow Conv")

    def use_layer(self, conv_layer: str = None, fc_layer: str = None, **kwargs):
        def activation_hook(module, inputs, outputs):
            self.activations = outputs

        self.capture[conv_layer] = Capture(self.model, conv_layer).capture_activations(activation_hook)

        layer_obj = getattr(self.model, fc_layer)
        self.weights = layer_obj.weight

    @classmethod
    def __tellem_function__(cls, args):
        cam = cls(args.model)
        kwargs = args.kwargs
        cam.use_layer(**kwargs)
        return cam.get_cam(**kwargs)


class GradCAM(ImplementationBase):
    """

    Official Paper:
        - https://arxiv.org/pdf/1610.02391.pdf
    Official Implementation:
        - [Lua] https://github.com/ramprs/grad-cam/
    Unofficial Implementations:
        - https://github.dev/jacobgil/pytorch-grad-cam
        - https://github.com/frgfm/torch-cam
        - https://keras.io/examples/vision/grad_cam/

    Args:
        ImplementationBase ([type]): [description]


    Useage:
        model = Model()
        grad_cam = GradCAM(model, loss_func)
        grad_cam.use_layer(layer_name)
        grad_cam_overlay = grad_cam(x, y)

    """

    def __init__(self, model: Model = None, loss_func: Callable[..., Any] = None, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)

        self.loss_func = loss_func
        self.capture = {}

        # variables to hold data
        self.activations = None
        self.grad = None

    def get_gradcam(self, x: Tensor, y: Tensor) -> Tensor:

        preds = self.model(x)

        loss = preds[:, y]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        grad = self.grad.squeeze(0)
        activations = self.activations.squeeze(0)

        # global average pool gradients over the width and height dimensions
        # and then unsqueeze for element multiplication
        grad = grad.mean(dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)

        cam_vals = grad * activations

        # sum along feature maps
        if cam_vals.ndim > 2:
            cam_vals = cam_vals.sum(dim=0)
        return cam_vals

    def use_layer(self, conv_layer: str):
        def gradient_hook(grad):
            self.grad = grad

        def activation_hook(module, inputs, outputs):
            outputs.register_hook(gradient_hook)
            self.activations = outputs

        self.capture[conv_layer] = Capture(self.model, conv_layer).capture_activations(activation_hook)
