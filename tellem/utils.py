import numpy as np
import PIL.Image as Image
import torch
from torchvision import transforms

from tellem.types import Tensor

_USES_TORCH = True


def noise_line(tensor):
    tensor_shape = tensor.shape
    return tensor + torch.randn(tensor.shape)


def upsample_to_image(image: Tensor, overlay: Tensor) -> Image:
    """
    used for CAM and various other things where you need to go from
    intermediate activations to image

    Args:
        overlay ([type]): [description]
        image ([type]): [description]
    """
    to_pil = transforms.ToPILImage()
    image, overlay = to_pil(image), to_pil(overlay)
    overlay = overlay.resize(image.size)
    return overlay


def clamp_image(x: Tensor, min: float = 0.0, max: float = 1.0) -> Tensor:
    return torch.clamp(x, min, max)
