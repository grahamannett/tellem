from tellem.implementations.base import ImplementationBase
from tellem.types import Tensor, Model

# https://github.com/Nguyen-Hoa/Activation-Maximization
# https://github.com/MisaOgura/flashtorch/blob/9422bf9dddfe7c65b13871b1c9f455e0a54f02dc/flashtorch/activmax/gradient_ascent.py#L272
# https://github.com/hans66hsu/nn_interpretability/blob/main/1.Activation_Maximization.ipynb
class Activation_Maximization(ImplementationBase):
    def __init__(self, model: Model):
        super().__init__()
