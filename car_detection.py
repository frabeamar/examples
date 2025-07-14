import json
from collections import defaultdict
from cProfile import label
from dataclasses import dataclass, fields
from pathlib import Path

import kornia
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import torch
import torch.nn as nn
from matplotlib import category
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F
from typing_extensions import Literal

from models import get_torchvision_model

colors = list(mcolors.CSS4_COLORS.keys())


import matplotlib.pyplot as plt
import skimage
from torch.utils.data import Dataset
from typer import Typer

app = Typer()


@dataclass
class Label:
    bbox: tuple[float, float, float, float]  # xmin, ymin, xlen, ylen
    image_id: int
    category_id: int
    area: float
    iscrowd: bool
    id: int


@dataclass
class Sample:
    img: torch.Tensor
    labels: list[Label]
    targets: dict


class CarDetectionDataset(Dataset):
    """
    A PyTorch Dataset for object detection with COCO-formatted annotations.
    """

    def __init__(
        self,
        root: Path,
        ann_file: Path,
    ) -> None:
        self.root = root
        text = json.loads(ann_file.read_text())
        self.images = pd.json_normalize(text, record_path=["images"])
        annotations = pd.json_normalize(text, record_path=["annotations"])
        self.categories = {
            c["id"]: c["name"] for c in json.loads(ann_file.read_text())["categories"]
        }

        self.annotations = annotations
        self.labels: dict[int, list[Label]] = defaultdict(list)
        for i in self.images["id"]:
            annot = annotations[annotations.image_id == i]
            self.labels[i] = [
                Label(**{key.name: a.get(key.name) for key in fields(Label)})
                for _, a in annot.iterrows()
            ]

        assert len(self.images) == len(self.labels)

    def convert_box_format(
        self,
        box: tuple[float, float, float, float],
        shape: tuple[int, int],
        to: Literal["XYHW", "XYXY"],
    ):
        """
        normalizes boxes to 1
        """
        a, b, c, d = box
        N, M = shape
        match to:
            case "XYHW":
                return (a, b, c - a, d - b)
            case "XYXY":
                return (a, b, a + c, b + d)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary containing the annotations.
        """
        img = self.images.iloc[index]
        # img = skimage.io.imread(self.root / img.file_name)
        image = read_image(self.root / img.file_name)

        label = self.labels[index]
        target = {}
        _, N, M = image.shape
        for l in label:
            l.bbox = self.convert_box_format(l.bbox, (N, M), "XYXY")

        target["boxes"] = tv_tensors.BoundingBoxes(
            [l.bbox for l in label],
            format="XYWH",
            canvas_size=F.get_size(image),
        )
        target["image_id"] = img.id
        # target["area"] = label.area
        # target["iscrowd"] = label.iscrowd

        return Sample(image, label, target)

    def __len__(self) -> int:
        return len(self.images)


def collate(batch: list[Sample]):
    img = torch.stack([s.img for s in batch])
    labels = [b.labels for b in batch]
    var = [
        {"boxes": torch.stack([torch.tensor(l.bbox) for l in img]), "labels": torch.tensor([l.category_id for l in img])}
        for img in labels
    ]
    return img / 255, var
    # labels, category_id


def visualize_sample(image: np.ndarray, targets: list[Label], i):
    fig, ax = plt.subplots()
    plt.imshow(image)

    for target in targets:
        box = target.bbox
        rect = plt.Rectangle(
            (box[0], box[1]),
            box[2],
            box[3],
            fill=False,
            edgecolor=colors[target.category_id],
            linewidth=2.5,
        )
        ax.add_patch(rect)
        label_name = target.category_id
        ax.text(
            box[0],
            box[1] - 10,
            f"{label_name}",
            bbox={"facecolor": colors[target.category_id], "alpha": 0.5, "pad": 1},
            color="white",
        )

    plt.savefig(f"sample_{i:05}.png")
    plt.close()


def visualize_classes(dataset: CarDetectionDataset):
    ids = dataset.annotations.category_id
    categories = dataset.categories
    data = dataset.annotations.assign(category=ids.apply(lambda x: categories[x]))
    sns.histplot(
        data,
        x="category",
    )
    plt.savefig("hist.png")


def train(dataset: CarDetectionDataset, num_epochs: int = 10):
    model, preprocess = get_torchvision_model()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        sampler=None,
        collate_fn=collate,
        # TODO: write a sampler
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    for i in range(num_epochs):
        for i, batch in enumerate(dataloader):
            images, boxes = batch
            images = preprocess(images)
            optimizer.zero_grad()
            losses = model(images, boxes)
            loss = sum([* losses.values()])
            loss.backward()
            print(loss)
            optimizer.step()
        torch.save(model.state_dict(), Path(f"./models/epoch_{i}"))

def evaluate(dataset: CarDetectionDataset, num_epochs: int = 10):
    model, preprocess = get_torchvision_model()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        sampler=None,
        collate_fn=collate,
        # TODO: write a sampler
    )
    model.eval()
    for i in range(num_epochs):
        for i, batch in enumerate(dataloader):
            images, boxes = batch
            images = preprocess(images)
            losses = model(images)
        torch.save(model.state_dict(), Path(f"./models/epoch_{i}"))




if __name__ == "__main__":
    # Instantiate dataset
    root_path = (
        Path(__file__).parent
        / "car_detection/data/Apply_Grayscale/Apply_Grayscale/Vehicles_Detection.v9i.coco/train"
    )
    dataset = CarDetectionDataset(
        root=root_path, ann_file=root_path / "_annotations.coco.json"
    )

    train(
        dataset,
    )
    visualize_classes(dataset)

    for i, (img, target) in enumerate(iter(dataset)):
        visualize_sample(img, target, i)
