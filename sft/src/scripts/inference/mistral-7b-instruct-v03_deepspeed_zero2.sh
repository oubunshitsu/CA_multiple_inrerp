fold=$1
checkpoint=$2
model_id="mistralai/Mistral-7B-Instruct-v0.3"
gpu_efficient_method="deepspeed_zero2"
dataset_style="conversational_style"
data_portion="test"

model_path="sft/results/${dataset_style}/${gpu_efficient_method}/${model_id}/${fold}/${checkpoint}"
test_data_path="sft/data/${dataset_style}/${fold}/${data_portion}.jsonl"
output_dir="${model_path}/generation"
# Should be a csv file 
output_file="${output_dir}/${data_portion}_outputs.csv"

mkdir -p $output_dir

uv run python sft/src/inference.py \
  --model_path_or_id $model_path \
  --attn_implementation "sdpa" \
  --tokenizer_id $model_id\
  --test_data_path $test_data_path\
  --output_file $output_file\

