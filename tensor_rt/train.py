import os
import tarfile
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import wget
from torchvision import datasets, models

# Define main data directory
DATA_DIR = Path.home() / "./data/imagenette2-320"
# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")


def download_data(DATA_DIR):
    """
    Downloads a subset of imagenet with 10 classes
    'tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball' and 'parachute'.
    """
    if os.path.exists(DATA_DIR):
        if not os.path.exists(os.path.join(DATA_DIR, "imagenette2-320")):
            url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
            wget.download(url)
            # open file
            file = tarfile.open("imagenette2-320.tgz")
            # extracting file
            file.extractall(DATA_DIR)
            file.close()
            os.remove("imagenette2-320.tgz")
    else:
        print("This directory doesn't exist. Create the directory and run again")


if not os.path.exists(Path.home() / "data"):
    os.mkdir(Path.home() / "data")
download_data(Path.home() / "data")


# This function allows you to set the all the parameters to not have gradients,
# allowing you to freeze the model and not undergo training during the train step.
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# Define functions for training, evalution, saving checkpoint and train parameter setting function
def train_epoch(
    model: nn.Module, dataloader: torchdata.DataLoader, crit, opt, epoch: int
):
    model.train()
    running_loss = 0.0
    for idx, (batch, labels) in enumerate(dataloader):
        batch, labels = batch.cuda(), labels.cuda(non_blocking=True)
        opt.zero_grad()
        out = model(batch)
        loss = crit(out, labels)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        if idx % 100 == 99:
            print(
                "Batch: [%5d | %5d] loss: %.3f"
                % (idx + 1, len(dataloader), running_loss / 100)
            )
            running_loss = 0.0


def dtype_to_torch_type(dtype: Literal["fp32", "fp16", "int8"]):
    match dtype:
        case "fp32":
            return torch.float32
        case "fp16":
            return torch.float16
        case "int8":
            return torch.int8
        case _:
            assert False


def evaluate(model, dataloader, crit, epoch: int, dtype: Literal["fp32", "fp16"] = "fp32"):
    total = 0
    correct = 0
    loss = 0.0
    class_probs = []
    class_preds = []
    model.eval()

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.cuda(), labels.cuda(non_blocking=True)
            out = model(data.to(dtype_to_torch_type(dtype)))
            loss += crit(out, labels)
            preds = torch.max(out, 1)[1]
            class_probs.append([F.softmax(i, dim=0) for i in out])
            class_preds.append(preds)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    evaluate_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    evaluate_preds = torch.cat(class_preds)

    return loss / total, correct / total


def save_checkpoint(state, ckpt_path="checkpoint.pth"):
    torch.save(state, ckpt_path)
    print("Checkpoint saved")


# Helper function to benchmark the model
def timeit(
    model,
    dtype: Literal["fp32", "fp16"],
    input_shape=(1024, 1, 32, 32),
    nwarmup=10,
    nruns=10,
):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda").to(dtype_to_torch_type(dtype))

    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            _ = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
    print("Average batch time: %.2f ms" % (np.mean(timings) * 1000))
    return np.mean(timings) * 1000


def dataset_splits():
    # Performing Transformations on the dataset and defining training and validation dataloaders
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    calib_dataset = torch.utils.data.random_split(val_dataset, [2901, 1024])[1]
    train_dataloader = torchdata.DataLoader(
        train_dataset, batch_size=32, shuffle=True, drop_last=True
    )
    val_dataloader = torchdata.DataLoader(
        val_dataset, batch_size=64, shuffle=False, drop_last=True
    )
    calib_dataloader = torchdata.DataLoader(
        calib_dataset, batch_size=64, shuffle=False, drop_last=True
    )
    return train_dataloader, val_dataloader, calib_dataloader


def load_model(feature_extract: bool):
    model = models.mobilenet_v2(pretrained=True)
    set_parameter_requires_grad(model, feature_extracting=feature_extract)
    model.classifier[1] = nn.Linear(1280, 10)
    model = model.cuda()
    return model


def benchmark(
    method_name: str, model, val_dataloader, criterion, dtype: Literal["fp32", "fp16"]
) -> dict:
    test_loss, test_acc = evaluate(model, val_dataloader, criterion, 0, dtype=dtype)
    timings = timeit(model, input_shape=(64, 3, 224, 224), dtype=dtype)
    print(f"{method_name}: {(test_acc * 100):.2f}%")
    return {
        "method": method_name,
        "time": timings,
        "acc": float(test_acc),
        "loss": float(test_loss),
    }
