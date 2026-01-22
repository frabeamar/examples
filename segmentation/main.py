from datasets import load_dataset
import torch
import cv2
import numpy as np
from PIL import Image
import io
from torchvision import transforms
import kornia as K
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt

import albumentations as A

# Define the pipeline
transform = A.Compose(
    [
        # Geometric: Flip and Rotate
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Morphological: Elastic deformations are great for cells
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        # Color: Staining variations
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        # Required for training
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def plot_examples(dataset: torch.utils.data.Dataset):
    for feat in ["image", "inst", "type", "sem"]:
        images = [
            Image.open(io.BytesIO(dataset[i][feat])).convert("RGB") for i in range(9)
        ]
        resize = transforms.Resize((224, 224))

        stacked = (
            np.array([resize(i) for i in images])
            .reshape((3, 3, 224, 224, 3))
            .transpose(0, 2, 1, 3, 4)
            .reshape(224 * 3, 224 * 3, 3)
        )
        if feat in {"sem", "type"}:
            stacked = stacked * 255
        cv2.imwrite(f"dataset_{feat}.png", stacked)


def from_bytes(data):
    # this is a batch
    for feat in ["image", "inst", "type", "sem"]:
        data[feat] = [Image.open(io.BytesIO(d)).convert("RGB") for d in data[feat]]
        data[feat] = [np.array(d) for d in data[feat]]

    return data


class PanopticSegmentationHead(nn.Module):
    def __init__(self, backbone, in_channels, n_tissue_classes, n_cell_types):
        super().__init__()

        # hack
        # take out the last layer
        backbone.conv = nn.Identity()
        self.backbone = backbone
        # 1. Semantic Head: Region classification (Tumor, Stroma, etc.)
        # Output: [B, n_tissue_classes, H, W]
        self.semantic_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, n_tissue_classes, kernel_size=1),
        )

        # 2. Instance Head: Distance Map (Predicts 'peaks' for cell centers)
        # Output: [B, 1, H, W] - Usually Linear/Tanh for regression
        self.instance_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
        )

        # 3. Cell Type Head: Specific cell classification (Lymphocyte, Cancer, etc.)
        # Output: [B, n_cell_types, H, W]
        self.type_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, n_cell_types, kernel_size=1),
        )

    def forward(self, x):
        x = self.backbone(x)
        sem_logits = self.semantic_head(x)
        inst_map = self.instance_head(x)  # Regression output (no softmax)
        type_logits = self.type_head(x)

        return {"sem": sem_logits, "inst": inst_map, "type": type_logits}


class HistologyDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        """
        Args:
            hf_dataset: The loaded Hugging Face dataset object.
            transform: PyTorch transforms for the image (RGB).
        """
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def transform_inst_to_dist(self, inst_mask: np.array):
        dist = distance_transform_edt(inst_mask)
        dist_map = dist / dist.max()

        return dist_map

    def __getitem__(self, idx):
        item = self.ds[idx]

        image = Image.open(io.BytesIO(item["image"])).convert("RGB")
        transformed = {}
        for feat in ["sem", "type", "inst"]:
            transformed[feat] = Image.open(io.BytesIO(item[feat]))
            transformed[feat] = np.array(transformed[feat])

        if self.transform:
            augmented = self.transform(
                image=np.array(image),
                sem=transformed["sem"],
                type=transformed["type"],
                inst=transformed["inst"],
            )
        image = augmented.pop("image")
        image = torch.from_numpy(image).permute((2, 0, 1))

        inst = augmented.pop("inst")
        inst = self.transform_inst_to_dist(inst)

        return {
            "image": image,
            "inst": inst,
            **transformed,
            "id": item.get("id", idx),
        }


def train_panoptic():
    train = load_dataset("histolytics-hub/panoptils_refined", split="train")
    train = HistologyDataset(train, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=2)
    unet = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=True,
    )
    criterion_sem = K.losses.DiceLoss()  # Semantic (Region)
    criterion_inst = nn.MSELoss()  # Instance (Distance Map)
    counts = torch.tensor([5000, 1200, 300])  # [Inflam, Stroma, Rare_Cell]
    criterion_type = nn.CrossEntropyLoss(weight=counts.sum() / counts)  # Cell Type
    model = PanopticSegmentationHead(unet, 32, 9, 9)
    optim = torch.optim.AdamW(model.parameters(), lr = 1e-5)
    for d in train_dataloader:
        optim.zero_grad()
        pred = model(d["image"])
        loss = (
            criterion_sem(pred["sem"], d["sem"])
            + criterion_inst(pred["inst"], d["inst"])
            + criterion_type(pred["type"], d["type"])
        )
        loss.backward()
        optim.step()

train_panoptic()
