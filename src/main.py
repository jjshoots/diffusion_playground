from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from wingman import Wingman

from diffusion_model import DiffusionModel
from utils import display_tensor_image, get_dataset


def get_model(wm: Wingman) -> DiffusionModel:
    # load the model
    model = DiffusionModel(
        input_shape=(1, 28, 28),
        sampling_steps=wm.cfg.sampling_steps,
        learning_rate=wm.cfg.learning_rate,
        device=wm.device,
        compile=False,
    )
    model = torch.compile(model)
    model = cast(DiffusionModel, model)

    # load in the weights
    has_weights, _, mark_dir = wm.get_weight_files()
    if has_weights:
        model.load_state_dict(
            torch.load(mark_dir / "weights.pth", map_location=wm.device),
        )

    return model


def train(wm: Wingman) -> None:
    # load the dataset
    trainloader = DataLoader(
        get_dataset(train=True),
        batch_size=wm.cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    # load the model
    model = get_model(wm)

    # begin training
    for epoch in tqdm(range(wm.cfg.epochs)):
        for images, labels in trainloader:
            # move things to device
            images = images.to(wm.device)
            labels = labels.to(wm.device)

            # update the model
            loss = model.update(images, labels)

            # record and checkpoint
            to_save, _, mark_dir = wm.checkpoint(loss=loss, step=epoch)
            if to_save:
                torch.save(model.state_dict(), mark_dir / "weights.pth")


@torch.no_grad()
def sample(wm: Wingman) -> None:
    # load the model
    model = get_model(wm)

    num_samples = 10
    noised_x = torch.randn((num_samples, 1, 28, 28), device=wm.device)
    blank_t = torch.ones((num_samples), dtype=torch.int64, device=wm.device)
    for t in reversed(range(1, wm.cfg.sampling_steps)):
        noised_x = model.reverse_diffusion_1_step(
            noised_x=noised_x,
            t=blank_t * t,
            c=torch.arange(num_samples, device=wm.device),
        )

        display_tensor_image(noised_x.detach().view(1, -1, 28))



if __name__ == "__main__":
    wm = Wingman(Path(__file__).parent.parent / "config.yaml")
    if wm.cfg.mode.train:
        train(wm)
    elif wm.cfg.mode.sample:
        sample(wm)
    else:
        print("Guess this is life now.")
