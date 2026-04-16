cd ..
git clone git@github.com:565353780/base-trainer.git

cd base-trainer
./dev_setup.sh

pip install timm einops diffusers flow_matching thop torchcfm

pip install cupy-cuda12x
