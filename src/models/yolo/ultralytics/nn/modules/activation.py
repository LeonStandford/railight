import torch
import torch.nn as nn


class AGLU(nn.Module):

    def __init__(self, device=None, dtype=None) -> None:
        super().__init__()
        self.act = nn.Softplus(beta=-1.0)
        self.lambd = nn.Parameter(
            nn.init.uniform_(torch.empty(1, device=device, dtype=dtype))
        )
        self.kappa = nn.Parameter(
            nn.init.uniform_(torch.empty(1, device=device, dtype=dtype))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam = torch.clamp(self.lambd, min=0.0001)
        return torch.exp(1 / lam * self.act(self.kappa * x - torch.log(lam)))
