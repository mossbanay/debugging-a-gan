import matplotlib.pyplot as plt

from collections import OrderedDict

import torch
import torchvision
from torch import nn, optim, autograd
from torchvision import utils

import pytorch_lightning as pl


class WGANGenerator(nn.Module):
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
                nn.ConvTranspose2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
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


class WGANCritic(nn.Module):
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
        for i in range(n_blocks-1):
            convs.extend(block((2**i) * n_filters))

        self.conv_blocks = nn.Sequential(
            *block(img_channels, n_filters, normalise=False),
            *convs,
        )

        ds_size = img_size // 2 ** (n_blocks)
        final_filters = (2**(n_blocks-1)) * n_filters

        self.adv_layer = nn.Sequential(
            nn.Linear(final_filters * ds_size * ds_size, 1),
        )

    def forward(self, img):
        """
        Takes a (batch_size, img_channels, img_size, img_size) generated or real images
        Returns a (batch_size, 1) tensor of probabilities that the input is real
        """
        out = self.conv_blocks(img)
        out = out.view(out.size(0), -1)

        return self.adv_layer(out)


class WGAN(pl.LightningModule):
    def __init__(self,
        latent_dim,
        img_size,
        args,
        output_img_path=None,
        img_channels=3):

        super().__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.lr = args.learning_rate
        self.c = args.weight_clip_thres
        self.gp_lambda = args.gp_lambda
        self.output_img_path = output_img_path
        self.use_gp = args.use_gp

        self.g = WGANGenerator(
            latent_dim=latent_dim,
            img_size=img_size,
            img_channels=img_channels,
            n_filters=args.n_filters,
            n_blocks=args.n_blocks,
        )

        self.d = WGANCritic(
            img_size=img_size,
            img_channels=img_channels,
            n_filters=args.n_filters,
            n_blocks=args.n_blocks,
        )

        self.output_z = torch.randn(16, self.latent_dim)
        self.epoch_n = 0

    def forward(self, x):
        return self.g(x)

    def gradient_penalty(self, real_img, fake_img):
        # Create random uniformly distributed weights
        eta = torch.zeros((real_img.size(0), 1, 1, 1)).type_as(real_img)
        eta.uniform_()

        # Create interpolated image
        interpolated_img = eta*real_img + (1-eta)*fake_img.detach()
        interpolated_img.requires_grad_(True)

        # Calculate the score of the interpolated image
        interpolated_score = self.d(interpolated_img)

        # Calculate the gradients of the score
        grads = autograd.grad(
            outputs=interpolated_score,
            inputs=interpolated_img,
            grad_outputs=torch.ones(real_img.size(0), 1).type_as(real_img),
            create_graph=True,
            retain_graph=True
        )[0]

        # Return the norm of the gradients times lambda
        return ((grads.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda

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
            # Compute the loss using the fake images and the labels
            fake_score = self.d(fake_img)
            
            # We want the score of the fake image to be as large as possible
            g_loss = -fake_score.mean()

            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict(
                {"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )

            return output

        # Train the critic
        elif optimizer_idx == 1:
            if not self.use_gp:
                # Clip weights in the critic to [-c, c]
                for p in self.d.parameters():
                    p.data.clamp(-self.c, self.c)

            # Compute the critic score on both the real and fake data
            fake_score = self.d(fake_img.detach())
            real_score = self.d(x)

            # Loss is equal to EM distance + gradient penalty (if enabled)
            if self.use_gp:
                d_loss = fake_score.mean() - real_score.mean() + self.gradient_penalty(x, fake_img)
            else:
                d_loss = fake_score.mean() - real_score.mean()

            tqdm_dict = {
                "d_loss": d_loss,
                "real_score": real_score.mean(),
                "fake_score": fake_score.mean(),
            }

            output = OrderedDict(
                {"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )

            return output

    def configure_optimizers(self):
        opt_g = optim.RMSprop(self.g.parameters(), lr=self.lr)
        opt_d = optim.RMSprop(self.d.parameters(), lr=self.lr)
        return [opt_g, opt_d], []

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx, second_order_closure):
        # Update the generator every 3 steps
        if optimizer_idx == 0:
            if batch_idx % 3 == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Update the critic every step
        if optimizer_idx == 1:
            optimizer.step()
            optimizer.zero_grad()

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
