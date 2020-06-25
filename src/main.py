import argparse
import os
import pytorch_lightning as pl
import torch

from pathlib import Path
from torchvision import datasets, transforms

from gan import GAN

N_TRAINING_SAMPLES = 10000
IMG_CHANNELS = 1
DATA_DIR = Path("~/.data/mnist").expanduser()
IMG_PATH = Path("./output_images/")


parser = argparse.ArgumentParser(description="Debugging a GAN")
parser = pl.Trainer.add_argparse_args(parser)

parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--learning-rate", default=1e-5, type=float)
parser.add_argument("--latent-dim", default=100, type=int)
parser.add_argument("--network", default="GAN", type=str)
parser.add_argument("--img-size", default=64, type=int)
parser.add_argument("--max-epochs", default=100, type=int)

args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize([args.img_size, args.img_size]),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5], std = [0.5])
])

DATA_DIR.mkdir(exist_ok=True, parents=True)

dataset = datasets.MNIST(DATA_DIR, transform=transform, download=True)
train_dataset = torch.utils.data.Subset(dataset, list(range(N_TRAINING_SAMPLES)))

kwargs = {"num_workers": 5, "pin_memory": True} if args.gpus else {}
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=0)
trainer = pl.Trainer.from_argparse_args(
    args,
    fast_dev_run=False,
    checkpoint_callback=checkpoint_callback,
)

v_num = trainer.logger.version
output_img_path = IMG_PATH / f'{v_num}'
output_img_path.mkdir(exist_ok=True, parents=True)

gan = GAN(
    latent_dim=args.latent_dim,
    img_size=args.img_size,
    img_channels=IMG_CHANNELS,
    output_img_path=output_img_path
)

trainer.fit(gan, train_loader)

os.system(f"ffmpeg -r 10 -i {output_img_path / f'real_imgs_%01d.png'} -vcodec libx264 -pix_fmt yuv420p -y {output_img_path / f'real_imgs.mp4'}")
os.system(f"ffmpeg -r 10 -i {output_img_path / f'gen_imgs_%01d.png'} -vcodec libx264 -pix_fmt yuv420p -y {output_img_path / f'gen_imgs.mp4'}")