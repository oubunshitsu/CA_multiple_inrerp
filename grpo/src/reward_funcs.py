import sys
import re
import evaluate
from typing import List


def reward_answer(
    completions,
    answer: List[str],
    **kwargs,
):
    """
    Reward answer regarding both format and correctness
    """
    rewards = []
    # completion_contents = completions
    # If the dataset is in conversational format
    completion_contents = [completion[0]["content"] for completion in completions]

    for completion, label_answer in zip(completion_contents, answer):
        reward = 0
        # Reward format
        answer_tag = r"<answer>(.*?)</answer>"
        matched_answers = re.findall(answer_tag, completion)
        # Reward the model if it generates exactly one tag
        if len(matched_answers) == 1:
            reward += 0.5

        # Reward correctness
        if len(matched_answers) > 0:
            generated_answer = matched_answers[0]
            if generated_answer.lower() == label_answer.lower():
                reward += 1

        rewards.append(reward)
    return rewards


def reward_slot(
    completions,
    slot: List[List[str]],
    **kwargs,
):
    """
    Reward slot for both format and rouge1
    max 1.5
    """
    rewards = []

    rouge = evaluate.load("rouge")

    # completion_contents = completions

    # If the dataset is in conversational format
    completion_contents = [completion[0]["content"] for completion in completions]

    for completion, label_slot in zip(completion_contents, slot):
        reward = 0
        # Reward format
        slot_tag = r"<slot>(.*?)</slot>"
        matched_slots = re.findall(slot_tag, completion)
        # Only reward slot if the current question asks for a slot
        if label_slot is not None:
            # Reward the model if it generates exactly one tag
            if len(matched_slots) == 1:
                reward += 0.5
            # Reward slot correctness only if the curent instance has a slot
            if len(matched_slots) > 0:
                generated_slot = matched_slots[0]
                # If the generated slot matches any of the label slots
                # The range of rouge1 is [0, 1]
                rouge_score = rouge.compute(
                    predictions=[generated_slot.lower()],
                    references=[[slot.lower() for slot in label_slot]],
                )["rouge1"]
                reward += float(rouge_score)
        else:
            # Penalize the model if it generates slot but current instance has no slot
            if len(matched_slots) > 0:
                reward -= 0.5

        rewards.append(reward)

    return rewards


def reward_all_func(
    prompts,
    completions,
    majority_answer: List[str],
    evidence: List[List[str]],
    slot: List[List[str]],
    **kwargs,
):
    """
    Reward everything in one function
    """
    rewards = []

    rouge = evaluate.load("rouge")

    # completion_contents = completions
    completion_contents = [completion[0]["content"] for completion in completions]

    for completion, label_answer, label_slot, label_evidence in zip(
        completion_contents, majority_answer, slot, evidence
    ):
        reward = 0
        ############# Reward format ################
        think_tag = r"<think>.*?</think>"
        answer_tag = r"<answer>.*?</answer>"
        slot_tag = r"<slot>.*?</slot>"

        matched_think_tags = re.findall(think_tag, completion)
        matched_answer_tags = re.findall(answer_tag, completion)
        matched_slot_tags = re.findall(slot_tag, completion)
        # Reward the model if it generates exactly one tag for each part
        if len(matched_think_tags) == 1:
            reward += 1
        else:
            # penalize the model if it generates more than 1 tag
            reward -= 1
        if len(matched_answer_tags) == 1:
            reward += 1
        else:
            reward -= 1

        if label_slot is not None:
            if len(matched_slot_tags) == 1:
                reward += 1
            else:
                reward -= 1
        else:
            # if model generates slot tags when the label is none
            if len(matched_slot_tags) > 0:
                reward -= 1
        ##############################################

        matched_thinkings = re.findall(r"<think>(.*?)</think>", completion)
        matched_answers = re.findall(r"<answer>(.*?)</answer>", completion)
        matched_slots = re.findall(r"<slot>(.*?)</slot>", completion)

        ################# Reward answer #################
        if len(matched_answers) > 0:
            generated_answer = matched_answers[0]
            if generated_answer.lower() == label_answer.lower():
                reward += 2
            else:
                reward -= 2
        else:
            # Be generous
            if label_answer.lower() in completion.lower():
                reward += 1
        ##################################################

        ################ Reward slot ####################
        if len(matched_slots) > 0:
            # If model generates a slot but label doesn't have a slot, penalize it
            if label_slot is None:
                reward -= 0.5
            else:
                generated_slot = matched_slots[0]
                # if the generated slot matches any of the label slots
                rouge_score = rouge.compute(
                    predictions=[generated_slot.lower()],
                    references=[[slot.lower() for slot in label_slot]],
                )["rouge1"]
                reward += float(rouge_score)
        ################################################

        ############### Reward thinking ################
        # Reward thinking based on the overlap between generation and selected evidence
        if len(matched_thinkings) > 0:
            if label_evidence is not None:
                generated_thinking = matched_thinkings[0]
                rouge_score = rouge.compute(
                    predictions=[generated_thinking.lower()],
                    references=[[evi.lower() for evi in label_evidence]],
                )["rougeL"]
                reward += float(rouge_score)
        else:
            reward -= 0.5

        ################################################

        rewards.append(reward)

    return rewards


if __name__ == "__main__":
    # test
    rewards = reward_all_func(
        prompts=["prompt"] * 2,
        completions=[
            "<think>bbb no evidence</think>abcidh<think>aaa</think><answer>yes</answer><slot>nnnn</slot>",
            "<think>no evidence</think><answer>yes</answer><slot>bbb</slot>",
        ],
        majority_answer=["yes"] * 2,
        slot=[["bbb"]] * 2,
        evidence=[["no evidence"]] * 2,
    )

    print(rewards)
