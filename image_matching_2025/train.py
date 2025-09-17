from collections.abc import Callable
from pathlib import Path
import random
from torch.utils.data import Dataset, default_collate
import skimage
import torch
from transformers import AutoImageProcessor
import pandas as pd
from transformers import SuperPointForKeypointDetection
import sys
import os
from tqdm import tqdm
from time import time, sleep
import gc
import numpy as np
import h5py
import dataclasses
import pandas as pd
from IPython.display import clear_output
from collections import defaultdict
from copy import deepcopy
from PIL import Image

import cv2
import torch
import torch.nn.functional as F
import kornia as K
import kornia.feature as KF

import torch
from lightglue import match_pair
from lightglue import ALIKED, LightGlue
from lightglue.utils import load_image, rbd
from transformers import AutoImageProcessor, AutoModel

# Utilities: importing data into colmap and competition metric
import pycolmap
sys.path.append('/kaggle/input/imc25-utils')
from database import *
from h5_to_db import *




class ImageMatchingDataset(Dataset):
    def __init__(self, label_file: Path, transform: Callable):
        self.labels = pd.read_csv(label_file)
        self.labels.rotation_matrix = self.labels.rotation_matrix.apply(
            lambda x: [float(d) for d in x.split(";")]
        )
        self.labels.translation_vector = self.labels.translation_vector.apply(
            lambda x: [float(d) for d in x.split(";")]
        )
        self.outliers = self.labels[self.labels["scene"] == "outliers"].reset_index(
            drop=True
        )
        self.labels = self.labels[self.labels["scene"] != "outliers"].reset_index(
            drop=True
        )
        self.image_path = label_file.parent / "train"
        self.transform = transform
        self.outliers_k = 5

    def __len__(self):
        return len(self.labels)

    def load_one(self, label_file, idx):
        img = label_file.iloc[idx]
        image = self.transform(
            torch.tensor(skimage.io.imread(self.image_path / img.dataset / img.image))
        ).pixel_values[0]
        return {
            "image": image,
            "rotation_matrix": torch.tensor(img.rotation_matrix),
            "translation_vector": torch.tensor(img.translation_vector),
        }

    def load_empty(self):
        return {
            "image": self.transform(torch.zeros((3, 224, 224))).pixel_values[0],
            "rotation_matrix": torch.zeros((3, 3)).flatten(),
            "translation_vector": torch.zeros((3,)),
        }

    def __getitem__(self, idx):
        main = self.load_one(self.labels, idx)
        scene = self.labels["scene"]
        in_scene = self.labels[scene == scene[idx]]
        from_scene = random.choices(in_scene.index, k=min(10, len(in_scene)))
        similar = [self.load_one(self.labels, i) for i in from_scene if i != idx]
        similar = similar + [self.load_empty()] * (10 - len(similar))
        assert len(similar) == 10

        outlier_paths = self.outliers[self.outliers.dataset == self.labels.dataset[idx]]
        outlier_subset = random.choices(
            outlier_paths.index, k=min(self.outliers_k, len(outlier_paths))
        )
        outliers = [self.load_one(self.outliers, i) for i in outlier_subset] + [
            self.load_empty()
        ] * (self.outliers_k - len(outlier_subset))
        assert len(outliers) == self.outliers_k
        return {
            "main": main,
            "similar": similar,
            "outliers": outliers,
            "scene": self.labels.scene[idx],
            "dataset": self.labels.dataset[idx],
        }


def collate(batch):
    return default_collate(batch)


def main():
    label_file = Path("/home/fmarsicano/data/image_matching_2025") / "train_labels.csv"
    processor, model = MODELS["superpoint"]
    dataset = ImageMatchingDataset(label_file, transform=processor)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate
    )

    model.to(torch.device("cuda"))
    for i, data in enumerate(dataloader):
        inputs = data["main"]["image"].to(torch.device("cuda"))
        out = model(inputs)
        breakpoint()


main()
