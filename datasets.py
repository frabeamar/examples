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
import torch
import kornia
import kornia.augmentation as K
import kornia.color as C
import kornia.filters as F
import kornia.geometry.transform as T
from PIL import Image
import torchvision.transforms as TV
import numpy as np

# Load image and convert to tensor
from abc import ABC

# Processing pipeline
from omegaconf import DictConfig, OmegaConf
import hydra

@dataclass
class TorchImage:
    image: torch.tensor

    @classmethod
    def from_path(self, path: Path):
        image = skimage.io.imread(path)
        image = kornia.utils.image_to_tensor(image).unsqueeze(0)
        return image / 255

    def as_image(self):
        return kornia.utils.tensor_to_image((self.image * 255).to(torch.uint8), keepdim=False)



@dataclass
class Pipeline(ABC):
    def __call__(self, x: TorchImage) -> TorchImage:
        pass


# Apply pipeline
@dataclass
class Augmentations(Pipeline):
    pipeline: K.AugmentationSequential

    @classmethod
    def from_config(cls, cfg: DictConfig):
        return cls(
            K.AugmentationSequential(
                K.RandomRotation(degrees=cfg.rotation),
                K.RandomResizedCrop(
                    size=tuple(cfg.random_resize_crop.size),
                    scale=cfg.random_resize_crop.scale,
                ),
                F.GaussianBlur2d(tuple(cfg.blur.kernel_size), tuple(cfg.blur.sigma)),
                data_keys=["input"],
                keepdim=True,
            )
        )

    def __call__(self, x: TorchImage):
        return x


@dataclass
class Transformer:
    pipeline: K.AugmentationSequential

    @classmethod
    def from_config(cls, cfg):
        return cls(
            K.AugmentationSequential(
                K.Resize(tuple(cfg.resize)),
                K.CenterCrop(
                    size=tuple(cfg.center_crop_size),
                ),
                # K.Normalize(tuple(cfg.normalize.mean), tuple(cfg.normalize.std)),
                data_keys=["input"],
                keepdim=True,
            )
        )

    def __call__(self, x: TorchImage) -> TorchImage:
        return self.pipeline(x)

class CustomDataset(Dataset):
    def __init__(self, cfg: DictConfig):
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
        image = TorchImage.from_path(img_path)
        #kornia adds a batch dimention
        image = self.augment(image)
        image = self.transform(image)

        return image[0]


# Example usage:
def show(x: TorchImage):
    breakpoint()
    skimage.io.imsave("here.png", TorchImage( x).as_image())


@hydra.main(version_base=None, config_path=".", config_name="config")
def app(cfg: DictConfig) -> None:
    dataset = CustomDataset(cfg.dataset)
    dataset[0]
    dataloader = DataLoader(dataset, **cfg.dataloader)
    for i, data in enumerate(dataloader):
        pass


if __name__ == "__main__":
    app()
