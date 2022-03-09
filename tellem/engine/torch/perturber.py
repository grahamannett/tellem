import torch


class Perturber:
    def __init__(
        self,
        dist: torch.distributions.distribution.Distribution = torch.distributions.normal.Normal(
            loc=0, scale=1.0
        ),
        num_perturbations: int = 64,
    ) -> None:
        self.dist = dist
        self.num_perturbations = num_perturbations

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._perturb(x).to(x.device)

    def _perturb(self, x: torch.Tensor) -> torch.Tensor:
        return self.dist.sample(x.shape)

    def repeat(self, x: torch.Tensor, _num_perturbations: int = None) -> torch.Tensor:
        _num_perturbations = _num_perturbations if _num_perturbations else self.num_perturbations
        x = x.repeat(_num_perturbations, *(1 for _ in x.shape))
        return x + self.__call__(x)
