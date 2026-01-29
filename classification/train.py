from pathlib import Path
from time import time

import timm
import torch
import torch.nn as nn
import torch.utils.tensorboard
import torchmetrics

# For video classification datasets, you might need torchvision.io or other libraries:
import torchmetrics.classification
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def plot_examples(dataset: torch.utils.data.Dataset):
    for i in range(81):
        img, label = dataset[i]
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(img.permute((1, 2, 0)).numpy())
        plt.savefig("dataset_example.png")
        plt.axis("off")
        ax.set_title(f"{label}")
    plt.savefig("dataset_example.png")


def timeit(fn):
    def wrapper():
        t = time()
        fn()
        return time() - t

    return wrapper


class ClassificationModel(nn.Module):
    def __init__(self, model_type: str, num_classes: int) -> None:
        super().__init__()

        self.feature_extractor = timm.create_model(
            model_type, pretrained=True, num_classes=0
        )
        self.dense = nn.Linear(self.feature_extractor.num_features, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.feature_extractor(x)
        x = self.dense(x)
        return x


def eval(
    eval_dataloader: DataLoader,
    model: ClassificationModel,
    metrics: torchmetrics.MetricCollection,
):
    model.eval()
    for data, label in tqdm.tqdm(
        eval_dataloader, "Evaluation", total=len(eval_dataloader)
    ):
        pred = model(data.cuda())
        metrics(pred, label.cuda())
    t = metrics.compute()
    return t


def train():
    data_root = Path.home() / "data/ml_datasets"
    dataset = CIFAR10(
        str(data_root),
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
        download=True,
    )
    train_tf = transforms.Compose(
        [
            # v2.ColorJitter(),
            transforms.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
        ]
    )
    eval_dataset, train_dataset = torch.utils.data.random_split(dataset, [0.1, 0.9])
    model_type = [
        "mobilenetv2_050",
        "resnet18",
    ]
    accs = []
    for m in model_type:
        model = ClassificationModel(m, 10).cuda()
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=128, shuffle=True
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True
        )
        writer = torch.utils.tensorboard.SummaryWriter(f"./logs/{m}")

        train_metric = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy("multiclass", num_classes=10, average="macro").cuda(),
                torchmetrics.Precision("multiclass", num_classes=10, average="macro").cuda(),
                torchmetrics.Recall("multiclass", num_classes=10, average="macro").cuda(),
                torchmetrics.F1Score("multiclass", num_classes=10, average="macro").cuda(),
            ]
        )
        eval_metric = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy("multiclass", num_classes=10, average="macro").cuda(),
                torchmetrics.Precision("multiclass", num_classes=10, average="macro").cuda(),
                torchmetrics.Recall("multiclass", num_classes=10, average="macro").cuda(),
                torchmetrics.F1Score("multiclass", num_classes=10, average="macro").cuda(),
            ]
        )
        optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
        lossfn = nn.CrossEntropyLoss()
        for epoch in range(50):
            model.train()
            for step, (data, label) in tqdm.tqdm(
                enumerate(train_dataloader), "Training", total=len(train_dataloader)
            ):
                optim.zero_grad()
                augmented = train_tf(data.cuda())
                pred = model(augmented)
                loss = lossfn(pred, label.cuda())

                loss.backward()
                optim.step()
                if step % 100 == 0:
                    with torch.no_grad():
                        train_metric(torch.argmax(pred, dim=1), label.cuda())
                        writer.add_scalar("training/loss", loss.item(), step)
                        writer.add_scalars(
                            "training/acc",
                            train_metric.compute(),
                            step + epoch * len(train_dataloader),
                        )
            with torch.no_grad():
                model.eval()
                eval(eval_dataloader, model, eval_metric)
                model.train()
                writer.add_scalars("eval/acc", train_metric.compute(), epoch)
        accs.append({"model": model_type, **train_metric.compute()})
        writer.close()


train()
