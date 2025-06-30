import os
import json
import torch
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from peft import PeftModel, PeftConfig
from dataclasses import dataclass, field
from tqdm import tqdm
from typing import List, Optional, Literal
import pandas as pd
import numpy as np


@dataclass
class Args:
    use_peft: bool = field(default=False)
    attn_implementation: Optional[str] = field(default=None)
    model_path_or_id: Optional[str] = field(
        default=None,
    )
    tokenizer_id: Optional[str] = field(
        default=None,
    )
    test_data_path: Optional[str] = field(
        default=None,
    )
    output_dir: Optional[str] = field(
        default=None,
    )
    device: Optional[str] = field(
        default="cuda",
    )
    temperature_lower_bound: Optional[float] = field(
        default=0.1,
    )
    temperature_upper_bound: Optional[float] = field(
        default=1.0,
    )


parser = HfArgumentParser(Args, allow_abbrev=False)
args = parser.parse_args_into_dataclasses()[0]

# load model
if args.use_peft:
    peft_config = PeftConfig.from_pretrained(args.model_path_or_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path
    )
    model = PeftModel.from_pretrained(base_model, args.model_path_or_id)
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path_or_id,
        attn_implementation=args.attn_implementation,
    )
model.eval()
model.to(args.device)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)

# For models that do not have a pad token
# consider eos token as the pad token
models_without_pad_token = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
if any(model_id in args.model_path_or_id for model_id in models_without_pad_token):
    tokenizer.pad_token = tokenizer.eos_token

# Generate temperatures from 0.1 to 1.0 avoiding floating points errors
num_points = (
    int(round((args.temperature_upper_bound - args.temperature_lower_bound) / 0.1)) + 1
)
temperatures = np.round(
    np.linspace(
        args.temperature_lower_bound, args.temperature_upper_bound, num=num_points
    ),
    1,
)

for temperature in temperatures:
    print(f"temperature: {temperature}")
    # Same params as what are used during training
    generation_config = GenerationConfig(
        max_new_tokens=256,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        top_p=1.0,
        top_k=50,
        min_p=None,
        repetition_penalty=1.0,
    )

    results = []
    with open(args.test_data_path, "r") as f:
        for line in tqdm(f):
            instance = json.loads(line)

            last_role = instance["prompt"][-1]["role"]
            if last_role == "user":
                add_generation_prompt = True
                continue_final_message = False
            elif last_role == "assistant":
                add_generation_prompt = False
                continue_final_message = True
            else:
                raise ValueError(f"Invalid role in the last message: {last_role}")
            prompt = tokenizer.apply_chat_template(
                instance["prompt"],
                continue_final_message=continue_final_message,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            prompt_inputs = tokenizer(
                text=prompt,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            prompt_ids, prompt_mask = (
                prompt_inputs["input_ids"].to(args.device),
                prompt_inputs["attention_mask"].to(args.device),
            )
            with torch.no_grad():
                prompt_completion_ids = model.generate(
                    prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=generation_config,
                )
            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            completions_text = tokenizer.batch_decode(
                completion_ids, skip_special_tokens=True
            )[0]
            instance.update({"generated_text": completions_text})
            results.append(instance)
    pd.DataFrame.from_records(results).to_csv(
        os.path.join(args.output_dir, f"testset_results_temperature_{temperature}.csv")
    )
