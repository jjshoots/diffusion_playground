import math
from typing import Literal, Sequence, cast

import torch
import torch.nn.functional as func
from torch import nn
from torch.cuda import CUDAGraph

from model import LinearModel
from utils import display_tensor_image


class DiffusionModel(nn.Module):
    """Diffusion."""

    def __init__(
        self,
        input_shape: Sequence[int],
        sampling_steps: int,
        learning_rate: float,
        device: torch.device | Literal["cpu", "cuda:0"],
        compile: bool,
    ) -> None:
        """__init__."""
        super().__init__()

        # some constants
        self._input_shape = input_shape
        self._sampling_steps = sampling_steps
        self._device = device

        # define the model and optimizer
        self._model = LinearModel(
            input_dim=math.prod(input_shape),
            timestep_dim=sampling_steps,
            context_dim=10,
        ).to(device)
        self._optim = torch.optim.Adam(
            self._model.parameters(),
            lr=learning_rate,
            capturable=True,
        )

        # precompute alphas
        t = torch.arange(sampling_steps, device=device)
        self._cum_alphas = torch.cos(torch.pi / 2.0 * t / self._sampling_steps) ** 2
        self._alphas = torch.zeros_like(self._cum_alphas)
        self._alphas[1:] = self._cum_alphas[1:] / self._cum_alphas[:-1]
        for _ in range(len(input_shape)):
            self._cum_alphas = self._cum_alphas.unsqueeze(-1)
            self._alphas = self._alphas.unsqueeze(-1)

        # for compilation
        if compile:
            self.compile()
        self.cuda_graph: None | torch.cuda.CUDAGraph = None
        self.loss_ref: None | torch.Tensor = None
        self.batch_ref: None | Sequence[torch.Tensor] = None

    def update(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> float:
        """update.

        Args:
            x (torch.Tensor): [B, ...] clean data
            c (torch.Tensor): [B] contextual information

        Returns:
            float: loss float
        """
        batch = [x, c]

        # for cudagraph running
        if self.cuda_graph is None:
            # make the batch ref, copy to batch ref
            self.batch_ref = [torch.zeros_like(s) for s in batch]
            [t.copy_(s) for s, t in zip([x, c], self.batch_ref)]

            # warmup teration
            self.forward(*self.batch_ref)  # pyright: ignore[reportArgumentType]

            # construct the graph
            torch.cuda.synchronize()
            self.cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cuda_graph):
                self.loss_ref = self.forward(*self.batch_ref)  # pyright: ignore[reportArgumentType]
            torch.cuda.synchronize()

        # cast some things so pyright doesn't scream
        self.cuda_graph = cast(CUDAGraph, self.cuda_graph)
        self.loss_ref = cast(torch.Tensor, self.loss_ref)
        self.batch_ref = cast(Sequence[torch.Tensor], self.batch_ref)

        # use the compiled graph
        torch.cuda.synchronize()
        self.train()
        [t.copy_(s) for s, t in zip(batch, self.batch_ref)]
        self.cuda_graph.replay()
        torch.cuda.synchronize()

        return self.loss_ref.cpu().numpy().item()

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """The main update function, wrapped in forward so we can compile."""
        # sample timesteps
        t = torch.randint(
            low=1,
            high=self._sampling_steps,
            size=(x.shape[0],),
            device=self._device,
        )

        # noise the images according to the timesteps
        noised_x, noise = self.forward_diffusion_n_step(x, t, c)

        # predict the noises from the noised images
        pred_noise_mean, pred_noise_var = self._model(noised_x, t, c)
        pred_noise = pred_noise_mean
        # pred_noise = torch.distributions.Normal(
        #     pred_noise_mean, pred_noise_var
        # ).rsample()

        # loss function, zero grad, backward, step
        loss = func.mse_loss(pred_noise, noise)
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        return loss.mean().detach()

    def forward_diffusion_n_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs n step of forward diffusion.

        Args:
            x (torch.Tensor): [B, ...] clean data
            t (torch.Tensor): [B] noise step
            c (torch.Tensor): [B] contextual information

        Returns:
            torch.Tensor: [B, ...] noised data
            torch.Tensor: [B, ...] the noise that made the data
        """
        # sample an absolute noise and an alpha
        noise = torch.randn_like(x)
        cum_alpha = self._cum_alphas[t]

        # noise the image
        noised_x = torch.sqrt(cum_alpha) * x + torch.sqrt(1 - cum_alpha) * noise

        return noised_x, noise

    def reverse_diffusion_1_step(
        self,
        noised_x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """reverse_diffusion_1_step.

        Performs n step of reverse diffusion.

        Args:
            noised_x (torch.Tensor): [B, ...] noised data
            t (torch.Tensor): [B] noise step
            c (torch.Tensor): [B] contextual information

        Returns:
            torch.Tensor: [B, ...] the input minus the predicted noise at the timestep
        """
        # get alphas
        cum_alpha = self._cum_alphas[t]
        alpha = self._alphas[t]

        # predict noise
        pred_noise_mean, pred_noise_var = self._model(noised_x, t, c)

        # reverse the process
        # fmt: off
        denoised_x = (
            (1 / torch.sqrt(alpha))
            * (noised_x - ((1 - alpha) / torch.sqrt(1 - cum_alpha)) * pred_noise_mean)
            # + torch.randn_like(noised_x) * pred_noise_var
        )
        # fmt: on

        denoised_x = torch.clamp(denoised_x, -1.0, 1.0)
        return denoised_x
