from datasets import load_dataset
import torch
import torchvision
from transformers import AutoImageProcessor
import matplotlib.pyplot  as plt
import cv2
import numpy as np
from PIL import Image
import io
from torchvision import transforms
def plot_examples(dataset: torch.utils.data.Dataset):
    for feat in ["image", "inst", "type", "sem"]:
        images = [Image.open(io.BytesIO(dataset[i][feat])).convert("RGB") for i in range(9)]
        resize = transforms.Resize((224, 224))
            
        stacked = np.array([resize(i) for i in images]).reshape( (3, 3, 224, 224, 3)).transpose(0, 2, 1, 3, 4).reshape(224*3, 224*3, 3)
        if feat in {"sem", "type"}:
            stacked  = stacked * 255
        cv2.imwrite( f"dataset_{feat}.png", stacked)

def from_bytes(data):
    # this is a batch
    for feat in ["image", "inst", "type", "sem"]:
        breakpoint()
        data[feat] = [Image.open(d).convert("RGB")for d in data[feat]]

    return data

train = load_dataset("histolytics-hub/panoptils_refined", split="train")
train.set_transform(from_bytes)
train.features
train_dataloader = torch.utils.data.DataLoader(train, shuffle=True)
for d in train_dataloader:
    pass
breakpoint()
test = load_dataset("histolytics-hub/panoptils_refined", split="test")
unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)

AutoImageProcessor.from_pretrained('mateuszbuda/brain-segmentation-pytorch', do_reduce_labels=True)
example_input = torch.randn(1, 3, 256, 256)
