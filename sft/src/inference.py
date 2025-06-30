import os
import warnings
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
from datasets import load_dataset
from trl.data_utils import is_conversational, apply_chat_template
# from trl.trainer.sft_trainer import _prepare_dataset


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
    output_file: Optional[str] = field(
        default=None,
    )
    device: Optional[str] = field(
        default="cuda",
    )


def get_args():
    parser = HfArgumentParser(Args, allow_abbrev=False)
    args = parser.parse_args_into_dataclasses()[0]
    return args


def load_model_and_tokenizer(args):
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

    return model, tokenizer


# # Same params as what are used during training
# generation_config = GenerationConfig(
#     max_new_tokens=256,
#     do_sample=True,
#     pad_token_id=tokenizer.pad_token_id,
#     bos_token_id=tokenizer.bos_token_id,
#     eos_token_id=tokenizer.eos_token_id,
#     temperature=temperature,
#     top_p=1.0,
#     top_k=50,
#     min_p=None,
#     repetition_penalty=1.0,
# )


def load_dataset_from_local_path(path):
    test_set = load_dataset(
        "json",
        data_files={"test": path},
        split="test",
    )
    return test_set


def prepare_input_for_conversational_data(example, tokenizer):
    # Delete the last message which is the assistant message which is the label for generation
    example["messages"] = example["messages"][:-1]

    tokenized = tokenizer.apply_chat_template(
        example["messages"], add_generation_prompt=True, return_tensors="pt"
    )
    example.update({"input_ids": tokenized})
    return example


def prepare_dataset(dataset, tokenizer):
    # Apply the chat template if needed
    first_example = next(iter(dataset))
    column_names = list(next(iter(dataset)).keys())
    if is_conversational(first_example):
        column_names = first_example.keys()
        # Remove the assistant message to create the input prompt
        dataset = dataset.map(
            prepare_input_for_conversational_data,
            fn_kwargs={"tokenizer": tokenizer},
            # remove_columns="messages"
            # if "messages" in column_names
            # else None,  # renamed to "text"
        )
        # See https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
    # When dataset is not conversational, we need to add the EOS token at the end of each example
    # We don't need to do this for conversational datasets as this is already handled by the
    # `apply_chat_template` function.
    else:

        def add_eos(example, eos_token):
            if "text" in example and not example["text"].endswith(
                eos_token
            ):  # language modeling case
                example["text"] = example["text"] + eos_token
            elif "completion" in example and not example["completion"].endswith(
                eos_token
            ):
                example["completion_with_eos"] = example["completion"] + eos_token
            return example

        dataset = dataset.map(
            add_eos,
            fn_kwargs={"eos_token": tokenizer.eos_token},
            remove_columns="messages"
            if "messages" in column_names
            else None,  # renamed to "text"
        )
        # Subsequent tokenization will add special tokens (mostly for bos).
        # See https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        add_special_tokens = True

        def tokenize(example, processing_class, dataset_text_field, add_special_tokens):
            if "prompt" in example:  # prompt-completion case
                processed_prompt = processing_class(
                    text=example["prompt"],
                    add_special_tokens=add_special_tokens,
                )
                processed = processing_class(
                    text=example["prompt"] + example["completion_with_eos"],
                    add_special_tokens=add_special_tokens,
                )

                # Check if the tokenized prompt starts with the tokenized prompt+completion
                prompt_ids = processed_prompt["input_ids"]
                prompt_completion_ids = processed["input_ids"]
                if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                    warnings.warn(
                        "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                        "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                        "token handling. Verify that the tokenizer is processing text consistently."
                    )

                # Create a completion mask
                completion_mask = [0] * len(prompt_ids) + [1] * (
                    len(prompt_completion_ids) - len(prompt_ids)
                )
                # processed = {**processed, **example, "completion_mask": completion_mask}
                processed = {**processed_prompt, **example}
            else:  # language modeling case
                processed = processing_class(
                    text=example[dataset_text_field],
                    add_special_tokens=add_special_tokens,
                )
                # Get all the key-value pairs in the input
                processed = {**processed, **example}
                # breakpoint()

            return processed

        dataset = dataset.map(
            tokenize,
            fn_kwargs={
                "processing_class": tokenizer,
                "dataset_text_field": "text",
                "add_special_tokens": add_special_tokens,
            },
        )
    return dataset


def main():
    args = get_args()
    model, tokenizer = load_model_and_tokenizer(args)
    test_set = load_dataset_from_local_path(args.test_data_path)
    test_set = prepare_dataset(test_set, tokenizer)

    results = []
    for instance in test_set:
        prompt_ids = torch.tensor(instance["input_ids"]).to(args.device)
        # attention_mask = torch.tensor([instance["attention_mask"]]).to(args.device)
        # breakpoint()
        with torch.no_grad():
            prompt_completion_ids = model.generate(
                prompt_ids,
                # attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False,
            )
        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completions_text = tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )[0]
        prompt_completion_text = tokenizer.batch_decode(
            prompt_completion_ids, skip_special_tokens=True
        )[0]
        for key in [
            "input_ids",
            "attention_mask",
            "completion_mask",
            "completion_with_eos",
        ]:
            if key in instance:
                del instance[key]
        instance.update({"generated_text_all": prompt_completion_text})
        instance.update({"generated_text": completions_text})
        results.append(instance)
        # break

    pd.DataFrame.from_records(results).to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
