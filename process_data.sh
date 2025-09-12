export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="/mnt/yuanyige/huggingface"
huggingface-cli login --token xx
export WANDB_BASE_URL=https://api.bandw.top
export WANDB_MODE=online
wandb login --relogin 0d370d89b0714775ee0b824f07f4c1f5ffa0b494

python /mnt/yuanyige/RLTest/verl-0.5.x/examples/data_preprocess/aime_xt.py