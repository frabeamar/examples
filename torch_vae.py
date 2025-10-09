from typing import Any
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
import tqdm

from tf_gan import BATCH_SIZE
from torch_gan import Reshape

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), 
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), 
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), 
            nn.Flatten(),
            nn.Linear(4096,  latent_dim),
        )
        self.mu = nn.Sequential(nn.Linear(latent_dim, 2), nn.Sigmoid())
        self.var = nn.Sequential(nn.Linear(latent_dim, 2), nn.ReLU())

        self.decoder = nn.Sequential(*[
            nn.Linear(2, 7*7*64),
            nn.ReLU(),
            Reshape((64, 7, 7)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), 
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), 
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        ])

    def forward(self,x) -> Any:
        latent = self.encoder(x)
        mu = self.mu(latent)
        var = self.var(latent)
        z = mu +torch.exp(-1/2*var) * torch.rand((BATCH_SIZE, 1))
        y =self.decoder(z)

        return z, mu, var, y


def kl_loss(mu, var):
    return   torch.mean(-0.5 * torch.sum(1 + var - mu.pow(2) - var.exp(), dim=-1))
    

def plot_images(model: VAE, epoch: int):
    noise = torch.rand((81, 2))
    images = model.decoder(noise).detach().cpu().permute([0, 2, 3, 1]).numpy()
    plt.figure(figsize=(9, 9))
    for i, image in enumerate(images):
        plt.subplot(9, 9, i + 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
    plt.savefig(f"./torch_vae_epoch_{epoch}")


if __name__ == "__main__":
    mse = torch.nn.BCELoss()
    vae = VAE(2)
    optim = torch.optim.Adam(vae.parameters(), lr=0.0001)
    dataset = FashionMNIST(root="./data", train=True, download=True,         transform=transforms.Compose(
            [transforms.ToTensor()]
        ),)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(30):
        pbar = tqdm.tqdm((dataloader))
        for ( data, y) in pbar:
            optim.zero_grad()
            z, mu, var, y = vae(data)
            kl = kl_loss(mu, var) 
            mean_loss =  mse(y, data)
            tot_loss = kl + 3*mean_loss
            tot_loss.backward()
            optim.step()
            pbar.set_postfix({"loss": tot_loss.item(), 
                              "kl":float(kl),
                              "mean":float(mean_loss),
                              "mu": float(mu.mean()), "var": float(var.mean())})
        plot_images(vae, epoch)
        
