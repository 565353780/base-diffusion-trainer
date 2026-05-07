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

`BaseTrainer` calls `preProcessDiffusionData` before `preProcessData`, so `BaseCFMTrainer` subclasses can override **`preProcessData`** directly for dataset-specific dtype / conditioning while still receiving flow-matching fields.

The model must accept a `dict` and return a `dict` with **`"v"`**; default loss is MSE on `v` vs ground-truth `data_dict["v"]`.
`BaseTrainer` also calls `getDiffusionLoss` before `getLossDict` and stores it as `result_dict["loss_diffusion"]`, so subclasses can override **`getLossDict`** directly while reusing the precomputed diffusion loss.

## Enjoy it~
