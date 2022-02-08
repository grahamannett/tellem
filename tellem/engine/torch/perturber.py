import torch


class Perturber:
    def __init__(
        self, dist: torch.distributions.distribution.Distribution = torch.distributions.normal.Normal(loc=0, scale=1.0)
    ) -> None:
        self.dist = dist

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.dist.sample(x.shape).to(x.device)

    def noise(self, x: torch.Tensor) -> torch.Tensor:
        pass
