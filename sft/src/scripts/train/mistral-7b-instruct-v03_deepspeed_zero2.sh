model_path_or_id="mistralai/Mistral-7B-Instruct-v0.3"
gpu_efficient_method="deepspeed_zero2"
# Can be either `prompt_completion_style` or `conversational_style`
dataset_style="conversational_style"
eval_steps=30
fold=$1

uv run deepspeed --num_gpus=8 sft/src/train.py \
  --model_path_or_id ${model_path_or_id} \
  --attn_implementation 'sdpa'\
  --data_dir "sft/data/${dataset_style}/${fold}"\
  --output_dir "sft/results/${dataset_style}/${gpu_efficient_method}/${model_path_or_id}/${fold}"\
  --wandb_project "logic_parsing_with_sft"\
  --wandb_group "sft"\
  --wandb_run_name "use_all_annotations_${dataset_style}_${model_path_or_id}_${gpu_efficient_method}_epoch1-60_${fold}"\
  --wandb_tags "sft" "all" ${gpu_efficient_method} ${fold} ${dataset_style}\
  --max_seq_length 2048 \
  --per_device_train_batch_size 8 \
  --optim "adamw_bnb_8bit" \
  --dataloader_num_workers 8\
  --torch_empty_cache_steps 8\
  --dataloader_pin_memory \
  --report_to "wandb"\
  --num_train_epochs 20 \
  --bf16 \
  --deepspeed "sft/src/deepspeed_config_zero2.json"\
  --eval_strategy "steps"\
  --eval_steps $eval_steps\
  --logging_steps $eval_steps\
  --save_steps $eval_steps\
  --overwrite_output_dir
  # --completion_only_loss true \
  # --resume_from_checkpoint 
  # --logging_strategy "epoch"\
  # --r 4\
  # --lora_alpha 4\
  # --use_lora \
  # --use_4bit_quantization
