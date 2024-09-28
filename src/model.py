import torch
from torch import nn


class DiffusionModel(nn.Module):
    """DiffusionModel."""

    def __init__(
        self,
        sampling_steps: int
    ) -> None:
        """__init__."""
        super().__init__()

        # some constants
        self.sampling_steps = sampling_steps

        # define the model
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
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
        # append the timestep embedding
        x = torch.concat(
            (
                x,
                (torch.ones_like(x) * t) - (self.sampling_steps / 2.0),
            ),
            dim=-3,
        )

        # pass input through model
        y = self.conv_layers(x)

        # split into mean, var
        noise_mean, noise_log_var = torch.split(y, 1, dim=-3)
        return noise_mean, noise_log_var

    def predict_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        mean, log_var = self(x, t)
        return torch.distributions.Normal(mean, torch.exp(log_var)).rsample()

    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """get_alpha.

        Args:
            t (torch.Tensor): [...] noise schedule in [0, 1].

        Returns:
            torch.Tensor: [...] alpha in [1, 0].
        """
        return self.get_cum_alpha(t) / self.get_cum_alpha(t - 1)

    def get_cum_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """get_alpha.

        Args:
            t (torch.Tensor): [...] noise schedule in [0, 1].

        Returns:
            torch.Tensor: [...] alpha in [1, 0].
        """
        t = t.clamp(min=0.0, max=self.sampling_steps)
        return torch.cos(torch.pi / 2.0 * t / self.sampling_steps) ** 2

    def forward_diffusion(
        self,
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
        alpha = self.get_alpha(t)

        # noise the image
        noised_x = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise

        return noised_x, noise

    def reverse_diffusion(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """reverse_diffusion.

        Args:
            x (torch.Tensor): [..., C, H, W] image.
            t (torch.Tensor): [..., 1] noise schedule in [0, 1].

        Returns:
            torch.Tensor: the input image minus the predicted noise at the timestep
        """
        # get alphas
        alpha = self.get_alpha(t)
        cum_alpha = self.get_cum_alpha(t)

        # predict noise
        pred_noise_mean, pred_noise_log_var = self(x, t)

        # reverse the process
        # fmt: off
        clean_x = (
            (1 / torch.sqrt(alpha))
            * (x - ((1 - alpha) / torch.sqrt(1 - cum_alpha)) * pred_noise_mean)
            + torch.randn_like(x) * torch.exp(pred_noise_log_var)
        )
        # fmt: on

        return clean_x
