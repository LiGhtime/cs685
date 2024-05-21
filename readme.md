# CS685 final project UMass Spring 2024

# Installing env for unsloth:

conda create --name unsloth_env python=3.10
conda activate unsloth_env

conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers

conda install jupyter

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

pip install --no-deps trl peft accelerate bitsandbytes
