import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
import torchmetrics
from torch.nn import (
    Module,
    ConvTranspose2d,
    Linear,
    Sequential,
    LeakyReLU,
    ReLU,
    Sigmoid,
    Conv2d,
    Flatten,
    Dropout,
)
import tqdm



DATA_DIR = Path.home() / "data"
NOISE_DIM = 128
BATCH_SIZE = 32


class Reshape(Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view((BATCH_SIZE,) + self.shape)


class PrintLayer(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x


class Generator(Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            *[
                Linear(NOISE_DIM, 7 * 7 * 256),
                Reshape((256, 7, 7)),
                ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
                LeakyReLU(),
                ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                LeakyReLU(),
                ConvTranspose2d(128, 1, 3, padding=1),
                Sigmoid(),
            ]
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


class Descriminator(Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Sequential(
            Conv2d(1, 256, 3, stride=2, padding=1),
            ReLU(),
            Conv2d(256, 128, 3, stride=2, padding=1),
            ReLU(),
            Flatten(),
            Linear(128 * 7 * 7, 64),
            Dropout(),
            Linear(64, 1),
            Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def plot_images(generator: Module):
    noise = torch.rand((81, NOISE_DIM))
    images = generator(noise)
    plt.figure(figsize=(9, 9))
    for i, image in enumerate(images):
        plt.subplot(9, 9, i + 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
    plt.show()



def main():
    dataset = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.137,), (0.387))]
        ),
    )
    gen = Generator()
    desc = Descriminator()

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    Goptimizer = torch.optim.Adam(gen.parameters(), lr=0.0001)
    Doptimizer = torch.optim.Adam(desc.parameters(), lr=0.0005)
    lossfn = torch.nn.BCELoss()
    Dmetrics = torchmetrics.Accuracy("binary")
    Gmetrics = torchmetrics.Accuracy("binary")
    pbar = tqdm.tqdm(train_loader)
    for epoch in range(20):
        Dmetrics.reset()
        Gmetrics.reset()
        for data, x_test in pbar:
            # gen step
            Goptimizer.zero_grad()
            fake = gen(torch.rand((BATCH_SIZE, NOISE_DIM)))

            y_pred = desc(fake)
            y_true = torch.ones((BATCH_SIZE, 1))
            g_loss = lossfn(y_pred, y_true)
            g_loss.backward()
            Goptimizer.step()
            Gmetrics.update(y_pred, y_true)

            # desc step
            Doptimizer.zero_grad()
            y_pred = desc(data)
            y_true = torch.concat(
                [torch.ones((BATCH_SIZE, 1)), torch.zeros((BATCH_SIZE, 1))]
            )

            d_loss = lossfn(y_pred, y_true)
            d_loss.backward()
            Doptimizer.step()
            Dmetrics.update(y_pred, y_true)
            pbar.set_postfix(
                {
                    "g_loss": float(g_loss),
                    "d_loss": float(d_loss),
                    "g_acc": Gmetrics.compute().item(),
                    "d_acc": Dmetrics.compute().item(),
                }
            )
        torch.save(gen.state_dict(), f"gen_{epoch}.pt")
        torch.save(desc.state_dict(), f"desc_{epoch}.pt")

        plot_images(gen)

if __name__ == "__main__":
    main()
