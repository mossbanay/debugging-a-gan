import matplotlib.pyplot as plt

from collections import OrderedDict

import torch
import torchvision
from torch import nn, optim, autograd
from torchvision import utils

import pytorch_lightning as pl


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim, img_size, img_channels=3, n_filters=16, n_blocks=3):
        super().__init__()

        self.n_filters = n_filters
        self.init_size = img_size // (2**n_blocks)

        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, n_filters * self.init_size * self.init_size)
        )

        def block(in_filters, out_filters=None):
            if out_filters is None:
                out_filters = 2*in_filters

            return [
                nn.BatchNorm2d(in_filters),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1),
            ]

        convs = []
        for i in range(n_blocks-1):
            convs.extend(block((2**i) * n_filters))

        self.conv_blocks = nn.Sequential(
            *convs,
            *block((2**(n_blocks-1)) * n_filters, img_channels),
            nn.Tanh(),
        )

    def forward(self, z):
        """
        Takes a (batch_size, latent_dim) tensor of noise
        Returns a (batch_size, img_channels, img_size, img_size) generated images
        """
        out = self.l1(z)
        out = out.view(out.size(0), self.n_filters, self.init_size, self.init_size)
        return self.conv_blocks(out)


class DCGANDiscriminator(nn.Module):
    def __init__(self, img_size, img_channels=3, n_filters=16, n_blocks=3):
        super().__init__()

        def block(in_filters, out_filters=None, normalise=True):
            if out_filters is None:
                out_filters = in_filters*2

            block = [
                nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]

            if normalise:
                block.append(nn.BatchNorm2d(out_filters, 0.8))

            return block

        convs = []
        for i in range(n_blocks):
            convs.extend(block((2**i) * n_filters))

        self.conv_blocks = nn.Sequential(
            *block(img_channels, n_filters, normalise=False),
            *convs,
        )

        ds_size = img_size // 2 ** (n_blocks+1)
        final_filters = (2**n_blocks) * n_filters

        self.adv_layer = nn.Sequential(
            nn.Linear(final_filters * ds_size * ds_size, 1), nn.Sigmoid()
        )

    def forward(self, img):
        """
        Takes a (batch_size, img_channels, img_size, img_size) generated or real images
        Returns a (batch_size, 1) tensor of probabilities that the input is real
        """
        out = self.conv_blocks(img)
        out = out.view(out.size(0), -1)

        return self.adv_layer(out)


class DCGAN(pl.LightningModule):
    def __init__(self,
        latent_dim,
        img_size,
        output_img_path=None,
        img_channels=3,
        lr=1e-4,
        n_filters=16,
        n_blocks=3):

        super().__init__()

        self.latent_dim = latent_dim
        self.lr = lr
        self.output_img_path = output_img_path

        self.g = DCGANGenerator(
            latent_dim=latent_dim,
            img_size=img_size,
            img_channels=img_channels,
            n_filters=n_filters,
            n_blocks=n_blocks
        )

        self.d = DCGANDiscriminator(
            img_size=img_size,
            img_channels=img_channels,
            n_filters=n_filters,
            n_blocks=n_blocks
        )

        self.output_z = torch.randn(16, self.latent_dim)
        self.epoch_n = 0

        self.update_cycle = 1

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

    def on_after_backward(self):
        if self.trainer.global_step % 28 == 0:
            params = self.state_dict()
            for name, grads in params.items():
                self.logger.experiment.add_histogram(tag=f'{name}_grad', values=grads, global_step=self.trainer.global_step)

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