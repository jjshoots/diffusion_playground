import torch
from torch import nn


class DiffusionModel(nn.Module):
    """DiffusionModel."""

    def __init__(self) -> None:
        """__init__."""
        super().__init__()

        self.model = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=5,
                    padding=2,
                ),
                nn.Conv2d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=5,
                    padding=2,
                ),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=16,
                    kernel_size=5,
                    padding=2,
                ),
                nn.Conv2d(
                    in_channels=16,
                    out_channels=2,
                    kernel_size=5,
                    padding=2,
                ),
            ],
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """forward.

        Args:
            x (torch.Tensor): [..., C, H, W] input image + noise.
            t (torch.Tensor): [..., 1] noise schedule in [0, 1].

        Returns:
            torch.Tensor: mean of predicted noise.
            torch.Tensor: var of predicted noise.
        """
        # pass input through model
        for f in self.model:
            x = x + f(x)

        # split into mean, var
        noise_mean, noise_var = torch.split(x, 1, dim=-3)
        return noise_mean, noise_var

    @staticmethod
    def get_alpha(t: torch.Tensor) -> torch.Tensor:
        """get_alpha.

        Args:
            t (torch.Tensor): [...] noise schedule in [0, 1].

        Returns:
            torch.Tensor: [...] alpha in [1, 0].
        """
        return DiffusionModel.get_cum_alpha(t) / DiffusionModel.get_cum_alpha(t - 1)

    @staticmethod
    def get_cum_alpha(t: torch.Tensor) -> torch.Tensor:
        """get_alpha.

        Args:
            t (torch.Tensor): [...] noise schedule in [0, 1].

        Returns:
            torch.Tensor: [...] alpha in [1, 0].
        """
        t = t.clamp(min=1e-2, max=1.0)
        return torch.cos(torch.pi / 2.0 * t) ** 2

    @staticmethod
    def forward_diffusion(
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A forward diffusion process.

        Args:
            x (torch.Tensor): [..., C, H, W] image.
            t (torch.Tensor): [..., 1] noise schedule in [0, 1].

        Returns:
            torch.Tensor: a noised image.
            torch.Tensor: the noise that contributed to the forward diffusion
        """
        # sample an absolute noise and an alpha
        noise = torch.randn_like(x)
        alpha = DiffusionModel.get_alpha(t)

        # noise the image
        noised_x = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise

        return noised_x, noise

    def reverse_diffusion(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # get alphas
        alpha = self.get_alpha(t)
        cum_alpha = self.get_cum_alpha(t)

        # predict noise
        pred_noise_mean, pred_noise_var = self(x, t)

        # reverse the process
        # fmt: off
        clean_x = (
            (1 / torch.sqrt(alpha))
            * (x - ((1 - alpha) / torch.sqrt(1 - cum_alpha)) * pred_noise_mean)
            + torch.randn_like(x) * pred_noise_var
        )
        # fmt: on

        return clean_x
