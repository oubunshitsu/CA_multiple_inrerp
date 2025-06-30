fold=$1
model_id="tiiuae/Falcon3-7B-Instruct"
output_dir="grpo/results/baseline/${model_id}/${fold}"

mkdir -p $output_dir

uv run python grpo/src/inference2.py \
  --model_path_or_id $model_id \
  --attn_implementation "sdpa" \
  --tokenizer_id $model_id\
  --test_data_path "grpo/data/${fold}/test.jsonl"\
  --output_dir $output_dir\
  --temperature_lower_bound 0.9\
  --temperature_upper_bound 0.9
