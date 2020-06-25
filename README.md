# Debugging a GAN

## Quick start guide

### Setting up the environment

To setup the environment, open a shell and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Training on a CPU

```bash
python src/main.py
```

### Training on a GPU

Pass the number of GPUs you want to use for training or -1 for all available GPUs.
You can also pass in training parameters (to see which add --help).

```bash
python src/main.py --gpus -1 --max-epochs 100
```

### Running tensorboard

To view the output in Tensorboard, run

```bash
tensorboard --logdir lightning_logs/
```

and go to http://localhost:6006.