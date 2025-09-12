set -x

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="/mnt/yuanyige/huggingface"
huggingface-cli login --token xx
export WANDB_BASE_URL=https://api.bandw.top
export WANDB_MODE=online
wandb login --relogin 0d370d89b0714775ee0b824f07f4c1f5ffa0b494


PROJECT_ROOT=/mnt/yuanyige/RLTest
LENGTH=8192
TBS=32
RO=8
MODEL=TTTXXX01/SFT_model
use_dynamic_bsz=true
num_gpu=4
aime24_train_path=$PROJECT_ROOT/data/aime24/train.parquet
aime24_test_path=$PROJECT_ROOT/data/aime24/train.parquet
aime25_train_path=$PROJECT_ROOT/data/aime25/train.parquet
aime25_test_path=$PROJECT_ROOT/data/aime25/train.parquet

train_files="['$aime24_train_path', '$aime25_train_path']"
test_files="['$aime24_test_path', '$aime25_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$TBS \
    data.max_prompt_length=1024 \
    data.max_response_length=$LENGTH \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((1024 + LENGTH)) \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TBS \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=null \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$RO \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=null \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='RLTest' \
    trainer.experiment_name='xt_aime2425' \
    trainer.n_gpus_per_node=$num_gpu \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=2 \
    trainer.total_epochs=1000000 $@
