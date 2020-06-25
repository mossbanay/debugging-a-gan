import matplotlib.pyplot as plt

from collections import OrderedDict

import torch
import torchvision
from torch import nn, optim, autograd
from torchvision import utils

import pytorch_lightning as pl


class GANGenerator(nn.Module):
    def __init__(self, latent_dim, img_size, img_channels=3, internal_dim=512):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.img_channels = img_channels
        self.internal_dim = internal_dim

        def block(in_dim, out_dim):
            return [
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
            ]

        self.model = nn.Sequential(
            *block(latent_dim, internal_dim),
            *block(internal_dim, internal_dim),
            *block(internal_dim, internal_dim),
            nn.Linear(internal_dim, img_size*img_size*img_channels),
            nn.Tanh(),
        )

    def forward(self, z):
        """
        Takes a (batch_size, latent_dim) tensor of noise
        Returns a (batch_size, img_channels, img_size, img_size) generated images
        """
        out = self.model(z)
        return out.view(z.size(0), self.img_channels, self.img_size, self.img_size)


class GANDiscriminator(nn.Module):
    def __init__(self, img_size, img_channels=3, internal_dim=512):
        super().__init__()

        self.img_size = img_size
        self.img_channels = img_channels
        self.internal_dim = internal_dim

        def block(in_dim, out_dim):
            return [
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
            ]

        self.model = nn.Sequential(
            nn.Flatten(),
            *block(img_size*img_size*img_channels, internal_dim),
            *block(internal_dim, internal_dim),
            *block(internal_dim, internal_dim),
            nn.Linear(internal_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        """
        Takes a (batch_size, img_channels, img_size, img_size) generated or real images
        Returns a (batch_size, 1) tensor of probabilities that the input is real
        """
        return self.model(img)


class GAN(pl.LightningModule):
    def __init__(self, latent_dim, img_size, output_img_path=None, img_channels=3, lr=1e-4):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.lr = lr
        self.output_img_path = output_img_path

        self.g = GANGenerator(latent_dim=latent_dim, img_size=img_size, img_channels=img_channels)
        self.d = GANDiscriminator(img_size=img_size, img_channels=img_channels)

        self.output_z = torch.randn(16, self.latent_dim)
        self.epoch_n = 0

        self.update_cycle = 1

    @staticmethod
    def create(img_size, img_channels):
        return GAN(
            latent_dim=512,
            img_size=img_size,
            img_channels=img_channels)

    def forward(self, x):
        return self.g(x)

    def adversarial_loss(self, y_hat, y):
        return nn.functional.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        batch_size = x.size(0)

        # Generate comparison images at the start of each epoch using real images
        if batch_idx == 0 and optimizer_idx == 0:
            self.plot_figs(x[0:16, :, :, :].detach())

        # Sample some noise
        z = torch.randn(batch_size, self.latent_dim)
        z = z.type_as(x)

        # Generate a fake image using the noise
        fake_img = self.g(z)

        # Train the generator
        if optimizer_idx == 0:
            # Create the labels as real (since we want the generator to produce realistic images)
            labels = torch.ones(batch_size, 1).type_as(x)

            # Compute the loss using the fake images and the labels
            g_prob = self.d(fake_img)
            g_loss = self.adversarial_loss(g_prob, labels)

            # Return the loss
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict(
                {"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )

            return output

        # Train the discriminator
        elif optimizer_idx == 1:
            # Create the labels for the real and fake images
            real_labels = torch.ones(batch_size, 1).type_as(x)
            fake_labels = torch.zeros(batch_size, 1).type_as(x)

            # Compute the loss for real and fake images
            real_loss = self.adversarial_loss(self.d(x), real_labels)
            fake_loss = self.adversarial_loss(self.d(fake_img), fake_labels)

            # Set the discriminator loss to be the average
            d_loss = (real_loss + fake_loss) / 2

            # Return the loss
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict(
                {"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )

            return output

    def configure_optimizers(self):
        opt_g = optim.Adam(self.g.parameters(), lr=self.lr, betas=(0.5, 0.99))
        opt_d = optim.Adam(self.d.parameters(), lr=self.lr, betas=(0.5, 0.99))
        return [opt_g, opt_d], []

    def plot_figs(self, real_imgs):
        """
        A simple helper function to log generated images throughout training
        """

        self.epoch_n += 1

        gen_imgs = self(self.output_z.type_as(real_imgs))

        grid = torchvision.utils.make_grid(0.5*real_imgs + 0.5, nrow=4)
        self.logger.experiment.add_image("real_imgs", grid, self.epoch_n)
        torchvision.utils.save_image(grid, self.output_img_path / f"real_imgs_{self.epoch_n}.png")

        grid = torchvision.utils.make_grid(0.5*gen_imgs + 0.5, nrow=4)
        self.logger.experiment.add_image("gen_imgs", grid, self.epoch_n)
        torchvision.utils.save_image(grid, self.output_img_path / f"gen_imgs_{self.epoch_n}.png")

        return