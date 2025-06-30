import pandas as pd
import numpy as np
import json
from transformers import AutoTokenizer


def to_instruction_data(processed_data_path, output_file_path):
    instruction_dataset = []
    with open(processed_data_path, "r") as f:
        for line in f:
            instance = json.loads(line)
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Given a pair of initial-argument and counter-argument, answer the question about the logic of the counter-argument in relation to the initial-argument.

### Input:
Initial-argument:
{instance["ia"]}

Counter-argument:
{instance["ca"]}

Question:
{instance["question"]}

### Response:
"""
            completion = instance["answer"]
            if completion == "yes":
                if instance["slot"] is not None:
                    completion += f". the thing is '{instance['slot']}'"

            instruction_dataset.append({"prompt": prompt, "completion": completion})
    pd.DataFrame.from_records(instruction_dataset).to_json(
        output_file_path,
        orient="records",
        lines=True,
    )


def to_reward_data(processed_data_path, output_file_path):
    reward_dataset = []
    with open(processed_data_path, "r") as f:
        for line in f:
            instance = json.loads(line)
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Given a pair of initial-argument and counter-argument, answer the question about the logic of the counter-argument in relation to the initial-argument. Your answer should strictly follow the format: 

1. Analyze the logic of the counter-argument and the question, think how the question can be answered based on the logic the counter-argument, generate your thoughts in between a <think></think> tag
2. Answer the question with 'yes' or 'no' enclosed in a <answer></answer> tag
3. If you are asked to extract a phrase from the counter-argument, generate the phrase and put it in a <slot></slot> tag, if the question doesn't require you to extract a phrase, ignore this step.

For example, a valid response would look like the following:
With an extracted phrase: <think>The counter-argument's logic is......thus, my answer is yes</think><answer>yes</answer><slot>slotfiller</slot>
Without an extracted phrase: <think>The counter-argument's logic is......thus, my answer is yes</think><answer>yes</answer>

### Input:
Initial-argument:
{instance["ia"]}

Counter-argument:
{instance["ca"]}

Question:
{instance["question"]}

### Response:
"""
            # print(prompt)
            # return
            reward_dataset.append({
                "prompt": prompt,
                "majority_answer": instance["majority_answer"],
                "evidence": instance["evidence"],
                "slot": instance["slot"],
            })
    pd.DataFrame.from_records(reward_dataset).to_json(
        output_file_path,
        orient="records",
        lines=True,
    )


def test_length(input_data_dir, style, tokenizer_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if style == "conversation":
        all_lengths = []
        for portion in ["train", "dev", "test"]:
            with open(
                f"{input_data_dir}/{portion}.jsonl",
                "r",
            ) as f:
                for line in f:
                    prompt = json.loads(line)["prompt"]
                    tokenized_prompt = tokenizer.apply_chat_template(
                        prompt, tokenize=True
                    )
                    all_lengths.append(len(tokenized_prompt))
        print(max(all_lengths))


def to_conversational_data(processed_data_path, output_file_path):
    conversational_dataset = []
    with open(processed_data_path, "r") as f:
        for line in f:
            instance = json.loads(line)
            sys_prompt = """Given a pair of initial-argument and counter-argument, answer the question about the logic of the counter-argument in relation to the initial-argument. Your answer should strictly follow the format: 

1. Answer the question with 'yes' or 'no' enclosed in a <answer></answer> tag
2. If you are asked to extract a phrase from the counter-argument, extract the phrase and put it in a <slot></slot> tag (Note that you should extract exactly the same phrase without paraphrasing from the CA if it exists), if the question doesn't require you to extract a phrase, ignore this step.

For example, a valid response would look like the following:
With an extracted phrase: <answer>yes</answer><slot>slotfiller</slot>
Without an extracted phrase: <answer>yes</answer>"""

            user_prompt = f"""Initial-argument:
{instance["ia"]}

Counter-argument:
{instance["ca"]}

Question:
{instance["question"]}"""
            # print(sys_prompt)
            # print(user_prompt)
            # break
            response = f"<answer>{instance['answer']}</answer>"
            if instance["slot"] is not None:
                slot = instance["slot"][0]
                response += f"<slot>{slot}</slot>"

            # conversational_dataset.append({
            #     "messages": [
            #         {"role": "system", "content": sys_prompt},
            #         {"role": "user", "content": user_prompt},
            #         {"role": "assistant", "content": response},
            #     ]
            # })
            instance["messages"] = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response},
            ]
            conversational_dataset.append(instance)
            # breakpoint()

    with open(output_file_path, "w") as f:
        for item in conversational_dataset:
            f.write(json.dumps(item) + "\n")


def to_prompt_completion_data(processed_data_path, output_file_path):
    prompt_completion_dataset = []
    with open(processed_data_path, "r") as f:
        for line in f:
            instance = json.loads(line)
            sys_prompt = """Given a pair of initial-argument and counter-argument, answer the question about the logic of the counter-argument in relation to the initial-argument. Your answer should strictly follow the format: 

1. Answer the question with 'yes' or 'no' enclosed in a <answer></answer> tag
2. If you are asked to extract a phrase from the counter-argument, extract the phrase and put it in a <slot></slot> tag (Note that you should extract exactly the same phrase without paraphrasing from the CA if it exists), if the question doesn't require you to extract a phrase, ignore this step.

For example, a valid response would look like the following:
With an extracted phrase: <answer>yes</answer><slot>slotfiller</slot>
Without an extracted phrase: <answer>yes</answer>"""

            user_prompt = f"""Initial-argument:
{instance["ia"]}

Counter-argument:
{instance["ca"]}

Question:
{instance["question"]}"""
            # print(sys_prompt)
            # print(user_prompt)
            # break
            prompt = sys_prompt + "\n\n" + user_prompt
            # breakpoint()

            completion = f"<answer>{instance['answer']}</answer>"
            if instance["slot"] is not None:
                slot = instance["slot"][0]
                completion += f"<slot>{slot}</slot>"

            # prompt_completion_dataset.append({
            #     "prompt": prompt,
            #     "completion": completion,
            # })
            instance["prompt"] = prompt
            instance["completion"] = completion
            prompt_completion_dataset.append(instance)
            # breakpoint()

    with open(output_file_path, "w") as f:
        for item in prompt_completion_dataset:
            f.write(json.dumps(item) + "\n")


def main():
    style = "conversational_style"
    for fold in range(3):
        for portion in ["train", "val", "test"]:
            to_conversational_data(
                processed_data_path=f"preprocessed_jsonl_data/fold{fold}/{portion}.jsonl",
                output_file_path=f"sft/data/{style}/fold{fold}/{portion}.jsonl",
            )
            # to_prompt_completion_data(
            #     processed_data_path=f"preprocessed_jsonl_data/fold{fold}/{portion}.jsonl",
            #     output_file_path=f"sft/{style}/fold{fold}/{portion}.jsonl",
            # )
        #     break
        # break


if __name__ == "__main__":
    main()
