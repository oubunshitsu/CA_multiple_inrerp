import os
import sys
import argparse
import re
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer
from dataclasses import dataclass, field
import wandb
import logging
import evaluate
from peft import LoraConfig, get_peft_model
from typing import List, Optional, Literal


#################### Args #######################
@dataclass
class CustomSFTConfig(SFTConfig):
    model_path_or_id: Optional[str] = field(
        metadata={"help": "Path to model files saved locally or a repo name"},
        default=None,
    )
    data_dir: Optional[str] = field(
        metadata={"help": "Path to the directory that contains all data files"},
        default=None,
    )
    wandb_project: Optional[str] = field(
        metadata={"help": "The project name to be used on wandb"}, default=None
    )
    wandb_group: Optional[str] = field(
        metadata={"help": "The group name used to group runs on wandb"}, default=None
    )
    wandb_run_name: Optional[str] = field(
        metadata={"help": "The name of the current run on wandb"}, default=None
    )
    wandb_tags: List[str] = field(
        metadata={"help": "The tags used for identifying different runs on wandb"},
        default_factory=list,
    )
    use_lora: bool = field(
        default=False, metadata={"help": "Whether to use lora or not"}
    )
    # Quantization config
    use_8bit_quantization: bool = field(default=False)
    use_4bit_quantization: bool = field(default=False)
    bnb_4bit_compute_dtype: Optional[str] = field(default=None)

    resume_from_checkpoint: bool = field(default=False)
    attn_implementation: Optional[str] = field(default=None)


# Workaround for using peft with HfArgumentParser
@dataclass
class CustomLoraConfig(LoraConfig):
    init_lora_weights: bool = field(default=True)
    layers_to_transform: Optional[int] = field(default=None)
    loftq_config: dict = field(default_factory=dict)
    trainable_token_indices: dict = field(default_factory=dict)
    # Comment out for llama models
    # target_modules: List[str] = field(default_factory=list)


parser = HfArgumentParser((CustomSFTConfig, CustomLoraConfig), allow_abbrev=False)
training_args, peft_args = parser.parse_args_into_dataclasses()

# print(training_args.resume_from_checkpoint)
# print(training_args.dataloader_pin_memory)
# print(training_args.bf16)
# print(training_args.gradient_checkpointing)
# print(training_args.wandb_tags)

# sys.exit(1)
# print("completion_only_loss: ", training_args.completion_only_loss)
print("bf16: ", training_args.bf16)

# Fix me for single-gpu training
local_rank = training_args.local_rank
if local_rank == 0:
    # Create wandb project only on the main process
    os.environ["WANDB_PROJECT"] = training_args.wandb_project
    wandb.init(
        project=training_args.wandb_project,
        group=training_args.wandb_group,
        name=training_args.wandb_run_name,
        tags=training_args.wandb_tags,
    )

# Load datasets
try:
    datasets = {}
    for portion in ["train", "val"]:
        file_path = f"{training_args.data_dir}/{portion}.jsonl"
        if os.path.exists(file_path):
            datasets[portion] = load_dataset(
                "json",
                data_files={f"{portion}": file_path},
                split=f"{portion}",
            )
except Exception as e:
    raise e
else:
    print(
        f"############ Loaded dataset: {list(datasets.keys())} from {training_args.data_dir} ################"
    )
train_dataset = datasets["train"] if "train" in datasets else None
eval_dataset = datasets["val"] if "val" in datasets else None


# Load model
# Quantization
if training_args.use_8bit_quantization:
    print(
        f"######### Load `{training_args.model_path_or_id}` with 8bit quantization ########"
    )
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        training_args.model_path_or_id,
        quantization_config=quantization_config,
        attn_implementation=training_args.attn_implementation,
    )
elif training_args.use_4bit_quantization:
    print(
        f"####### Load `{training_args.model_path_or_id}` with 4bit quantization #######"
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=training_args.bnb_4bit_compute_dtype
    )
    model = AutoModelForCausalLM.from_pretrained(
        training_args.model_path_or_id,
        quantization_config=quantization_config,
        attn_implementation=training_args.attn_implementation,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        training_args.model_path_or_id,
        attn_implementation=training_args.attn_implementation,
    )
# Peft
if training_args.use_lora:
    print(
        f"###### Using LoRA with r: {peft_args.r}, lora_alpha: {peft_args.lora_alpha}, target_modules: {peft_args.target_modules} #####"
    )
    # To be compatible with using LoRA
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_args)
    model.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained(training_args.model_path_or_id)

# For models that do not have a pad token
# consider eos token as the pad token
models_without_pad_token = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
if any(
    model_id in training_args.model_path_or_id for model_id in models_without_pad_token
):
    tokenizer.pad_token = tokenizer.eos_token

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# For logging purpose
if training_args.deepspeed is not None:
    print(
        f"####### Train model with deepspeed config at {training_args.deepspeed} #######"
    )

trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
