#  Implementing an explainability method

To implement a new methodology you will build off the `ImplementationBase` class and use the `Capture` class as well.  For this example we will be implementing **CAM** (_Class Activation Mapping_)  as it shows how you go about capturing activations but to capture values such as the gradient is only marginally more complicated.

The base class and constructor will just take a model

```python
from tellem import Capture
from tellem.implementations import ImplementationBase

class CAM(ImplementationBase):

    def __init__(self, model: Model = None, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.check_model()

        self.capture = {}
```

Since CAM applies to a specific model architecture we will also use a `check_model` method that ensures the model follows an architecture like: conv feature maps → global average pooling → softmax layer


```python
def check_model(self) -> None:
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
```

from here we explicitly say which layer to use for the CAM but this could also be set automatically by the class to use the last linear layer and then the preceding convolutional output.  In general it is preferred to explicitly tell the implementations which classes to use unless the method applies to all layers.

```python
def use_layer(self, conv_layer: str = None, fc_layer: str = None, **kwargs) -> None:
    def activation_hook(module, inputs, outputs):
        self.activations = outputs

    self.capture[conv_layer] = Capture(self.model, conv_layer).capture_activations(activation_hook)

    layer_obj = getattr(self.model, fc_layer)
    self.weights = layer_obj.weight
```

in this method you can see we define a function `activation_hook` that sets a property on the instance for the values of the activations.  This is technically unnecessary as the `Capture` class does something similar if you do not pass it a method for the `capture_activations` and `capture_gradients` methods but will be attached as properties on the `Capture` instance in those cases.  Here we also create a reference to the weights we want to use for our method.

From here we now need to show how to actually get the CAM for an input `x` and its label `y`.

```python
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
```

This method creates the activations by passing the input to the model and then grabs the activations and reshapes them as we are passing in a single sample.  We then must reshape the matrices as the weights and activations may be different due to how the number of dimensions of these matrices differs for frameworks such as pytorch.  Then the values are multiplied together and returned.  This would give you the CAM values you would then be able to overlay on the original image.

One last bit, as you may have seen we pass `**kwargs` to all the methods used.  To help automate tests, we can implement another method:

```python
@classmethod
def __tellem_function__(cls, args):
    cam = cls(args.model)
    kwargs = args.kwargs
    cam.use_layer(**kwargs)
    return cam.get_cam(**kwargs)
```

This method shows how to explicitly use the method and allows our testing framework to automatically test the class with minimal more overhead.




<!-- # Implementing Tests -->


<!-- # Implementing an attack


The first part of implementing a
 -->

