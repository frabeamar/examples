from pydantic import BaseModel
from typing import List

class RandomResizeCrop(BaseModel):
    size: List[int]
    scale: List[float]

class Blur(BaseModel):
    kernel_size: List[int]
    sigma: List[float]

class AugmentationConfig(BaseModel):
    rotation: int
    random_resize_crop: RandomResizeCrop
    blur: Blur

class Normalize(BaseModel):
    mean: List[float]
    std: List[float]

class TransformConfig(BaseModel):
    resize: List[int]
    center_crop_size: List[int]
    normalize: Normalize

class DatasetConfig(BaseModel):
    image_dir: str
    augmentations: AugmentationConfig
    transform: TransformConfig

class DataloaderConfig(BaseModel):
    batch_size: int
    num_workers: int
    shuffle: bool

class Config(BaseModel):
    dataset: DatasetConfig
    dataloader: DataloaderConfig
