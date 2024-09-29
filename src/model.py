import torch
from torch import nn
from torch.nn import functional as F


class LinearModel(nn.Module):
    """A normal neural network that predicts noise mean and variance."""

    def __init__(
        self,
        input_dim: int,
        timestep_dim: int,
        context_dim: int,
    ) -> None:
        """__init__.

        Args:

        Returns:
            None:
        """
        super().__init__()

        # some constants
        self.timestep_dim = timestep_dim
        self.context_dim = context_dim

        # define the model
        self.layer = nn.Sequential(
            nn.Linear(
                input_dim + timestep_dim + context_dim,
                2048,
            ),
            nn.ReLU(),
            nn.Linear(
                2048,
                input_dim * 2,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """forward.

        Args:
            x (torch.Tensor): [B, ...] clean data
            t (torch.Tensor): [B, ...] noise step
            c (torch.Tensor): [B, ...] contextual information

        Returns:
            torch.Tensor: mean of predicted noise.
            torch.Tensor: var of predicted noise.
        """
        # record input shape
        input_shape = x.shape

        # create encodings
        timestep_encoding = F.one_hot(t, num_classes=self.timestep_dim)
        context_encoding= F.one_hot(c, num_classes=self.context_dim)

        # concat the noise to the input
        x = torch.concat(
            (x.view(x.shape[0], -1), timestep_encoding, context_encoding),
            dim=-1,
        )

        # pass input through model
        y = self.layer(x)

        # split into mean, var
        noise_mean, noise_log_var = torch.split(y, y.shape[-1] // 2, dim=-1)
        noise_var = F.softplus(noise_log_var)

        # reshape and return
        noise_mean = noise_mean.view(input_shape)
        noise_var = noise_var.view(input_shape)
        return noise_mean, noise_var
