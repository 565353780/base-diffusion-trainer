cd ..
git clone https://github.com/565353780/base-trainer.git

cd base-trainer
./setup.sh

pip install timm einops diffusers flow_matching thop torchcfm

pip install cupy-cuda12x
