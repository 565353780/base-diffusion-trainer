cd ..
git clone git@github.com:565353780/base-trainer.git

cd base-trainer
./dev_setup.sh

pip install -U timm einops diffusers flow_matching thop torchcfm

pip install -U cupy-cuda12x
