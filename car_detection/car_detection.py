from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib.colors as mcolors
colors = list(mcolors.CSS4_COLORS.keys())


import skimage
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from typer import Typer


app = Typer()


@dataclass
class Label:
    bbox: tuple[float, float, float, float] # xmin, ymin, xlen, ylen
    image_id: int
    category_id: int
    area: float
    iscrowd: bool
    id: int


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
        self.annotations = annotations
        self.labels = defaultdict(list)
        for i in self.images["id"]:
            annot = annotations[annotations.image_id == i]
            self.labels[i] = [
                Label(**{key.name: a.get(key.name) for key in fields(Label)})
                for _, a in annot.iterrows()
            ]
        assert len(self.images) == len(self.labels)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary containing the annotations.
        """
        img = self.images.iloc[index]
        img = skimage.io.imread(self.root / img.file_name)
        return img, self.labels[index]

    def __len__(self) -> int:
        return len(self.images)


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


if __name__ == "__main__":
    # Instantiate dataset
    root_path = (
        Path(__file__).parent
        / "data/Apply_Grayscale/Apply_Grayscale/Vehicles_Detection.v9i.coco/train"
    )
    dataset = CarDetectionDataset(
        root=root_path, ann_file=root_path / "_annotations.coco.json"
    )

    for i, (img, target) in iter(enumerate(dataset)):
        visualize_sample(img, target, i)
