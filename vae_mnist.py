# vae_mnist.py
"""
Run:
    python vae_mnist.py
This trains a small VAE on MNIST, saves checkpoints, and writes sample images to disk.
"""
import os
import math
import random
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt

# --------------------
# Config / hyperparams
# --------------------
seed = 42
batch_size = 128
epochs = 20
lr = 1e-3
latent_dim = 32
hidden_dim = 512
log_interval = 100
save_dir = Path("vae_outputs")
save_dir.mkdir(exist_ok=True, parents=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed)
random.seed(seed)

# --------------------
# Data
# --------------------
transform = transforms.Compose([
    transforms.ToTensor(),                # 0..1
    # for BCE loss we want the inputs in [0,1]; for MSE we could normalize differently
])

train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# --------------------
# Model: simple fully-connected VAE
# --------------------
class VAE(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=512, latent_dim=32):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc_dec1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_dec2 = nn.Linear(hidden_dim, input_dim)
        # Activation
        self.act = nn.ReLU()

    def encode(self, x):
        # x: (batch, 1, 28, 28) or (batch, 784)
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        h = self.act(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.act(self.fc_dec1(z))
        x_recon = torch.sigmoid(self.fc_dec2(h))  # outputs in (0,1) for BCE
        x_recon = x_recon.view(-1, 1, 28, 28)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# --------------------
# Loss
# --------------------
def vae_loss(recon_x, x, mu, logvar, kld_weight=1.0):
    # recon_x, x: (B,1,28,28), values in [0,1]
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')  # sum over batch
    # KL divergence between N(mu, sigma^2) and N(0,1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + kld_weight * KLD, BCE, KLD

# --------------------
# Utilities: save samples and interpolations
# --------------------
def save_sample_grid(model, epoch, n=64, path=save_dir / "samples"):
    model.eval()
    os.makedirs(path, exist_ok=True)
    with torch.no_grad():
        z = torch.randn(n, latent_dim, device=device)
        samples = model.decode(z)  # (n,1,28,28)
        grid = utils.make_grid(samples, nrow=8, padding=2)
        utils.save_image(grid, path / f"sample_epoch_{epoch:03d}.png")
    model.train()

def save_reconstructions(model, data_loader, epoch, n=8, path=save_dir / "recons"):
    model.eval()
    os.makedirs(path, exist_ok=True)
    imgs = []
    recons = []
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(data_loader):
            x = x.to(device)
            recon, _, _ = model(x)
            imgs.append(x[:n].cpu())
            recons.append(recon[:n].cpu())
            break
    # stack original and recon side by side
    comp = torch.cat([imgs[0], recons[0]])
    grid = utils.make_grid(comp, nrow=n)
    utils.save_image(grid, path / f"recon_epoch_{epoch:03d}.png")
    model.train()

def interpolate(model, z1, z2, steps=8):
    model.eval()
    zs = [(z1 * (1 - t) + z2 * t) for t in torch.linspace(0, 1, steps, device=z1.device)]
    zs = torch.stack(zs, dim=0)
    with torch.no_grad():
        imgs = model.decode(zs).cpu()
    model.train()
    return imgs

# --------------------
# Train / Eval loops
# --------------------
if __name__ == '__main__':
    model = VAE(input_dim=28*28, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scaler = torch.cuda.amp.GradScaler()  # for mixed precision

    global_step = 0
    best_loss = 1e9

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                recon_batch, mu, logvar = model(data)
                loss, bce, kld = vae_loss(recon_batch, data, mu, logvar)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            global_step += 1

            if global_step % log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{train_loss / ((batch_idx + 1) * batch_size):.4f}",
                    "bce": f"{bce.item() / batch_size:.4f}",
                    "kld": f"{kld.item() / batch_size:.4f}"
                })

        avg_train_loss = train_loss / len(train_loader.dataset)
        print(f"====> Epoch: {epoch} Average loss: {avg_train_loss:.4f}")

        # Evaluate on test set (single pass)
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon, mu, logvar = model(data)
                loss, bce, kld = vae_loss(recon, data, mu, logvar)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader.dataset)
        print(f"====> Test set loss: {avg_test_loss:.4f}")

        # Save samples and reconstructions
        save_sample_grid(model, epoch)
        save_reconstructions(model, test_loader, epoch)

        # Checkpointing
        ckpt = save_dir / f"vae_epoch_{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "latent_dim": latent_dim,
        }, ckpt)

    print("Training complete. Samples and reconstructions saved to", save_dir)
