import torch

from pathlib import Path
from torchvision import datasets, transforms


def get_transforms(img_size, img_channels):
    return transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5]*img_channels, std = [0.5]*img_channels)
    ])

MNIST_DATA_DIR = Path("~/.data/mnist").expanduser()
def load_mnist(img_size, n_training_samples=10000):
    transform = get_transforms(img_size, 1)
    MNIST_DATA_DIR.mkdir(exist_ok=True, parents=True)
    mnist_dataset = datasets.MNIST(MNIST_DATA_DIR, transform=transform, download=True)
    return torch.utils.data.Subset(mnist_dataset, list(range(n_training_samples)))

POKEMON_SPRITE_PATH = Path('~/.data/pokemon_sprites').expanduser()
def download_pokemon_data():
    print("Downloading pokemon sprites ...")

    import requests
    from bs4 import BeautifulSoup
    resp = requests.get("https://pokemondb.net/sprites")
    soup = BeautifulSoup(resp.text, features='lxml')

    for gen_idx, gen_list in enumerate(soup.find_all(class_='infocard-list-pkmn-sm')):
        gen_idx += 1

        print(f"Downloading generation {gen_idx}")

        for pkmn in gen_list.find_all('a'):
            pkmn_src = pkmn.find('span')['data-src']
            pkmn_name = pkmn_src.split('/')[-1][:-4]

            output_path = POKEMON_SPRITE_PATH / f'{gen_idx}/{pkmn_name}.png'
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.exists():
                continue

            with requests.get(pkmn_src, stream=True) as r:
                r.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        f.write(chunk)

def load_pokemon(img_size):
    if not POKEMON_SPRITE_PATH.exists():
        download_data()

    print("Loading data ...")

    transform = get_transforms(img_size, 3)

    return datasets.ImageFolder(
        root=POKEMON_SPRITE_PATH,
        transform=transform
    )