# Base Diffusion Trainer

## Setup

```bash
conda create -n bdt python=3.10
conda activate bdt
./setup.sh
```

## Run

```bash
python demo.py
```

`BaseCFMTrainer` only assumes the model accepts one `dict` and returns one `dict`.
Align custom input/output keys in your `BaseCFMTrainer` subclass.

## Enjoy it~
