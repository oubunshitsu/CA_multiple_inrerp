model_path_or_id="tiiuae/Falcon3-7B-Instruct"
gpu_efficient_method="deepspeed_zero2"
num_generations=8
fold=$1

uv run deepspeed --num_gpus=8 grpo/src/train.py \
  --model_path_or_id ${model_path_or_id} \
  --attn_implementation 'sdpa'\
  --data_dir "grpo/data/${fold}"\
  --output_dir "grpo/results/${gpu_efficient_method}/${model_path_or_id}/${num_generations}/${fold}"\
  --wandb_project "logic_parsing_with_rl"\
  --wandb_group "grpo"\
  --wandb_run_name "use_all_annotations_reward_separately_${model_path_or_id}_${gpu_efficient_method}_generations-${num_generations}_epoch1-60_${fold}"\
  --wandb_tags "grpo" "all" ${gpu_efficient_method} ${fold} \
  --max_prompt_length 1024\
  --max_completion_length 1024\
  --per_device_train_batch_size $num_generations \
  --optim "adamw_bnb_8bit" \
  --dataloader_num_workers 8\
  --torch_empty_cache_steps 8\
  --dataloader_pin_memory \
  --num_generations $num_generations \
  --report_to "wandb"\
  --num_train_epochs 60 \
  --bf16 \
  --deepspeed "grpo/src/deepspeed_config_zero2.json"\
  --eval_strategy "steps"\
  --overwrite_output_dir
  # --resume_from_checkpoint 
  # --logging_strategy "epoch"\
  # --r 4\
  # --lora_alpha 4\
  # --use_lora \
  # --use_4bit_quantization

