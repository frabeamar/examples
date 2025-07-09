import matplotlib.pyplot as plt
import skimage
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pathlib import Path
from dataclasses import dataclass
from torchvision import transforms
import torch.nn as nn
import numpy as np

# Load image and convert to tensor
from abc import ABC

# Processing pipeline
from omegaconf import DictConfig, OmegaConf
import hydra
from config import (
    AugmentationConfig,
    Config,
    DatasetConfig,
    TransformConfig,
)


@dataclass
class NumpyImage:
    image: np.ndarray

    @classmethod
    def from_path(cls, path: Path):
        image = skimage.io.imread(path)
        return cls(image=image)


@dataclass
class Pipeline(ABC):
    def __call__(self, x: NumpyImage) -> NumpyImage:
        pass


# Apply pipeline
@dataclass
class Augmentations(Pipeline):
    pipeline: transforms.Compose

    @classmethod
    def from_config(cls, cfg: AugmentationConfig):
        return cls(
            transforms.Compose(
                [
                    transforms.RandomRotation(degrees=cfg.rotation),
                    transforms.RandomResizedCrop(
                        size=tuple(cfg.random_resize_crop.size),
                        scale=tuple(cfg.random_resize_crop.scale),
                        antialias=True,
                    ),
                    transforms.GaussianBlur(
                        kernel_size=tuple(cfg.blur.kernel_size),
                        sigma=tuple(cfg.blur.sigma),
                    ),
                ]
            )
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x.image = self.pipeline(x)
        return x


@dataclass
class Transformer:
    pipeline: transforms.Compose

    @classmethod
    def from_config(cls, cfg: TransformConfig):
        return cls(
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(tuple(cfg.resize)),
                    transforms.CenterCrop(
                        size=tuple(cfg.center_crop_size),
                    ),
                    # transforms.Normalize(tuple(cfg.normalize.mean), tuple(cfg.normalize.std)),
                ]
            )
        )

    def __call__(self, x: NumpyImage) -> torch.Tensor:
        return self.pipeline(x.image)


class CustomDataset(Dataset):
    def __init__(self, cfg: DatasetConfig):
        """
        Args:
            label_dict (dict): Dictionary mapping image filenames to labels.
            transform (callable, optional): Transform applied to all images (e.g., ToTensor, normalization).
            augment (callable, optional): Augmentation transforms applied ONLY during training.
        """
        self.image_dir = Path(cfg.image_dir)
        self.transform = Transformer.from_config(cfg.transform)
        self.augment = Augmentations.from_config(cfg.augmentations)

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        img_path = self.image_dir / f"img_{idx:05}.jpeg"
        image = NumpyImage.from_path(img_path)
        image = self.transform(
            image)
        image = self.augment(image)

        return image


# Example usage:
def show(x: torch.tensor):
    y = x.permute([1, 2, 0]) * 255 
    z = y.to(torch.uint8)
    skimage.io.imsave("here.png", z)


@hydra.main(version_base=None, config_path=".", config_name="config")
def app(cfg: DictConfig) -> None:
    pydantic_config = Config(**OmegaConf.to_container(cfg, resolve=True))
    dataset = CustomDataset(pydantic_config.dataset)
    dataloader = DataLoader(dataset, **pydantic_config.dataloader.model_dump())
    for i, data in enumerate(dataloader):
        show(data[0])


if __name__ == "__main__":
    app()
