import os
import time
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import numpy as np


class OpenAIModel:
    def __init__(self, model_id) -> None:
        self._client = OpenAI()
        self._model = model_id

    def call(self, sys_prompt: str, user_prompt: str):
        response = self._client.responses.create(
            model=self._model,
            input=[
                {
                    "role": "system",
                    "content": sys_prompt,
                },
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ca_logic_analysis",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string"},
                            "answer": {"type": "string"},
                            "slot": {"type": "string"},
                        },
                        "required": ["reason", "answer", "slot"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        )
        structured_output = json.loads(response.output_text)
        return structured_output


def main():
    # model_id = "gpt-4o-2024-08-06"
    # model_id = "gpt-4.1-2025-04-14"
    model_id = "o4-mini-2025-04-16"
    model = OpenAIModel(model_id=model_id)
    dataset_style = "conversational_style"

    for fold in ["fold0", "fold1", "fold2"]:
        print(f"Working on {fold}")
        test_data_path = f"prompt_engineering/data/{fold}/test.jsonl"
        output_dir = f"prompt_engineering/results/{dataset_style}/{fold}/{model_id}"

        results = []
        with open(test_data_path, "r") as f:
            total_lines = sum(1 for _ in f)
        with open(
            test_data_path,
            "r",
        ) as f:
            for line in tqdm(f, total=total_lines):
                instance = json.loads(line)
                while True:
                    try:
                        response = model.call(
                            sys_prompt=instance["prompt"][0]["content"],
                            user_prompt=instance["prompt"][1]["content"],
                        )
                    except Exception as e:
                        print(e)
                        time.sleep(5)
                    else:
                        break
                instance["response"] = response
                results.append(instance)
                # break

        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame.from_records(results).to_csv(
            os.path.join(output_dir, "testset_results.csv"),
            index=False,
        )


if __name__ == "__main__":
    main()
