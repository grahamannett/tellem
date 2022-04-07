from collections import defaultdict
from typing import Any, Dict, Generator, Hashable

import numpy as np
import PIL.Image as Image
import torch
from torchvision import transforms

_USES_TORCH = True


def noise_line(tensor):
    tensor_shape = tensor.shape
    return tensor + torch.randn(tensor.shape)


def upsample_to_image(image: torch.Tensor, overlay: torch.Tensor) -> Image:
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


def clamp_image(x: torch.Tensor, min: float = 0.0, max: float = 1.0) -> torch.Tensor:
    return torch.clamp(x, min, max)


class EasyDict:
    """similar to EasyDict package

    usage:

    obj = EasyDict(val=val1, args={...args})
    """

    def __init__(self, **kwargs) -> None:
        for key, val in kwargs.items():
            self.__setattr__(key, val)

    def __iter__(self) -> Generator[int, Hashable, Any]:
        for key, val in self.__dict__.items():
            yield key, val

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]


class NestedDefaultDict(defaultdict):
    """Similar to:
        rec_dd = lambda: defaultdict(rec_dd)
    which is a recursive defaultdict.
    Allows for easily setting keys with many layers.

    ===
    Not a great datastructure in many ways since you will fetch keys that dont exist if you use it incorrectly and wont
    get errors but still is useful when setting multiple nested dicts for other stuff
    ===


    Args:
        defaultdict (_type_): _description_
    """

    def __init__(self, *args, **kwargs) -> None:
        super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self) -> str:
        return repr(dict(self))

    def to_dict(self) -> Dict[Any, Any]:
        """converts to pure dictionary which is useful when saving/loading or so you dont access keys that it would
        otherwise automatically create
        call this if you want to save/load it later but you lose the nested functionality.

        Returns:
            Dict[Any, Any]: _description_
        """
        obj = dict(self)
        for key, val in obj.items():
            if isinstance(val, NestedDefaultDict):
                obj[key] = val.to_dict()
        return obj
