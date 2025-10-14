import tensorboard
from pathlib import Path
from torch.utils.data import DataLoader
from time import time
from cv2 import transform
from matplotlib import pyplot as plt
import torch.nn as nn
import torch
import matplotlib
import pandas as pd
from sympy import I, deg
import torch.utils.tensorboard
import torchmetrics
from torchvision.datasets import (
    # LSUNClass, no module lmdb
    CIFAR10,
    CIFAR100,
    EMNIST,
    FashionMNIST,
    QMNIST,
    MNIST,
    KMNIST,
    MovingMNIST,
    StanfordCars,
    STL10,
    SUN397,
    SVHN,
    PhotoTour,
    SEMEION,
    Omniglot,
    Flowers102,
    ImageNet,
    Caltech101,
    Caltech256,
    CelebA,
    USPS,
    Food101,
    DTD,
    FER2013,
    GTSRB,
    CLEVRClassification,
    OxfordIIITPet,
    PCAM,
    Country211,
    FGVCAircraft,
    EuroSAT,
    RenderedSST2,
    Imagenette
)
import torch
# For video classification datasets, you might need torchvision.io or other libraries:
from torchvision.datasets import Kinetics, HMDB51, UCF101
import  torchvision.transforms as transforms
import timm
import tqdm
import torchvision.transforms.v2 as v2

def plot_examples(dataset: torch.utils.data.Dataset):
    for i in range(81):
        img, label = dataset[i]
        ax = plt.subplot(3, 3, i+1)
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
    def __init__(self, model_type:str, num_classes: int) -> None:
        super().__init__()
        
        self.feature_extractor = timm.create_model(model_type, pretrained=True, num_classes=0)
        self.dense = nn.Linear(1536, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.feature_extractor(x)
        x = self.dense(x)
        return x

def eval(eval_dataloader:DataLoader, model:ClassificationModel):
    acc = torchmetrics.Accuracy("multiclass", num_classes=10).cuda()
    model.eval()
    for data, label in tqdm.tqdm(eval_dataloader):
        pred = model(data.cuda())
        acc(pred, label.cuda())
    t = acc.compute()
    return t

def train():
    data_root = Path.home() / "data/ml_datasets"
    dataset = CIFAR10(str(data_root), transform=transforms.Compose([
                                                 transforms.ToTensor(), 
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
                                              
                                                 ]), 
                                                 download=True)
    train_tf= transforms.Compose([   v2.ColorJitter(),
                                                 v2.GaussianBlur(1, 0.5), 
                                                 v2.RandomCrop((28, 28)),
                                                 v2.RandomRotation(degrees=10), 
                                                 ])
    eval_dataset, train_dataset = torch.utils.data.random_split(dataset, [0.1, 0.9])
    model_type = ["tf_efficientnet_b3", "resnetv2_101", "resnet_v2_50", "mobilevitv2_050", "mobilevitv2_100"]
    for m in model_type:
        model = ClassificationModel(m, 10).cuda()
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=32, shuffle=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        writer = torch.utils.tensorboard.SummaryWriter(f"./logs/{m}")
        acc = torchmetrics.Accuracy("multiclass", num_classes=10).cuda()
    
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        lossfn = nn.CrossEntropyLoss()
        for epoch in range(10):
            model.train()
            for step, (data, label) in tqdm.tqdm(enumerate(train_dataloader)):
                optim.zero_grad()
                augmented = train_tf(data.cuda())
                pred = model(augmented)
                loss = lossfn(pred, label.cuda())
                loss.backward()
                optim.step()
                if step %100 ==0:
                    acc(torch.argmax(pred, dim=1), label.cuda())
                    print(acc.compute())
                    writer.add_scalar("training/loss", loss.item(), step)
                    writer.add_scalar("training/acc", acc.compute().item(), step)
            metrics = eval(eval_dataloader, model)
            writer.add_scalar("eval/acc", metrics, epoch)
        writer.close()

train()
