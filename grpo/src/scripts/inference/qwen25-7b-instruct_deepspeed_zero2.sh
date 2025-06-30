fold=$1
checkpoint=$2
model_id="Qwen/Qwen2.5-7B-Instruct"
gpu_efficient_method="deepspeed_zero2"
num_generations=8
model_path="grpo/results/${gpu_efficient_method}/${model_id}/${num_generations}/${fold}/${checkpoint}"
output_dir="${model_path}/testset_outputs"


mkdir -p $output_dir

uv run python grpo/src/inference2.py \
  --model_path_or_id $model_path \
  --attn_implementation "sdpa" \
  --tokenizer_id $model_id\
  --test_data_path "grpo/data/${fold}/test.jsonl"\
  --output_dir $output_dir\
  --temperature_lower_bound 0.9\
  --temperature_upper_bound 0.9
