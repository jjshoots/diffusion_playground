from pathlib import Path
from typing import cast
from tqdm import tqdm

import torch
import torch.optim as opt
from torch.utils.data import DataLoader
from wingman import Wingman

from model import DiffusionModel
from utils import display_tensor_image, get_dataset


def main() -> None:
    wm = Wingman(Path(__file__).parent.parent / "config.yaml")

    # load the dataset
    trainloader = DataLoader(
        get_dataset(train=True),
        batch_size=wm.cfg.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # load the model, optimizer, loss_fn
    model = DiffusionModel(
        sampling_steps=wm.cfg.sampling_steps,
    ).to(wm.device)
    optim = opt.AdamW(
        model.parameters(),
        lr=wm.cfg.learning_rate,
    )
    loss_fn = torch.nn.MSELoss()

    # compile
    model = torch.compile(model)
    model = cast(DiffusionModel, model)

    # begin training
    for epoch in range(wm.cfg.epochs):
        for images, labels in tqdm(trainloader):
            images = images.to(wm.device)
            labels = labels.to(wm.device)

            # sample timesteps
            ts = torch.randint(
                low=1,
                high=wm.cfg.sampling_steps + 1,
                size=(len(images), 1, 1, 1),
                device=wm.device,
            )

            # noise the images according to a schedule
            noised_images, noises = model.forward_diffusion(images, ts)

            # perform reverse diffusion
            predicted_noises = model.predict_noise(noised_images, ts)

            # loss function, zero grad, backward, step
            loss = loss_fn(predicted_noises, noises)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # record and checkpoint
            to_save, model_dir, mark_dir = wm.checkpoint(
                loss=loss.detach().cpu().numpy(),
                step=epoch
            )
            torch.save(model.state_dict(), mark_dir / "weights.pth")


if __name__ == "__main__":
    main()
