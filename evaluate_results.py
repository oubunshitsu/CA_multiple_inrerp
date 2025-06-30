import os
import re
import pandas as pd
import evaluate
from tqdm import tqdm
from typing import Dict
from pathlib import Path
import statistics
from collections import defaultdict


def eval_answer_predicate_level(df, metric_name):
    # Get the list of labels and predictions as 1/0
    references = df.apply(
        lambda row: 0 if row["answer"] == "no" else 1, axis=1
    ).tolist()
    predictions = df.apply(
        lambda row: 0 if row["predicted_answer"] == "no" else 1, axis=1
    ).tolist()

    # Different metrics
    if metric_name == "acc":
        num_correct = 0
        for i, pred in enumerate(predictions):
            if pred == references[i]:
                num_correct += 1
        metric_result = num_correct / len(predictions)
    elif metric_name == "macro_f1":
        f1_metric = evaluate.load("f1")
        metric_result = f1_metric.compute(
            predictions=predictions, references=references, average="macro"
        )["f1"]

    # breakpoint()
    return metric_result


def eval_slot_predicate_level(df, metric_name):
    supported_metrics = ["rouge", "bertscore"]
    supported_coverages = ["all", "r"]

    metric_name, coverage = metric_name.split("_")

    assert metric_name in supported_metrics
    assert coverage in supported_coverages

    if coverage == "all":
        # Consider average on all instances, if label doesn't have a slot, prediction shouldn't have a slot either
        # consider slot = "null" if the current instance doesn't have a slot in the predicate or doesn't have a
        # slot in prediction
        df.slot = df.apply(
            lambda row: eval(row["slot"]) if isinstance(row.slot, str) else row["slot"],
            axis=1,
        )
        # All slots including the non-slots
        references = df.apply(
            lambda row: row["slot"] if isinstance(row.slot, list) else "null", axis=1
        ).tolist()
        predictions = df.apply(
            lambda row: row["predicted_slot"]
            if isinstance(row.predicted_slot, str)
            else "null",
            axis=1,
        ).tolist()
    elif coverage == "r":
        # Consider average only on the instances where the label has slot
        references = []
        predictions = []
        for _, row in df.iterrows():
            label_slot = row["slot"]
            pred_slot = row["predicted_slot"]
            if isinstance(label_slot, str) or isinstance(label_slot, list):
                references.append(
                    eval(label_slot) if isinstance(label_slot, str) else label_slot
                )

                predictions.append(pred_slot if pred_slot is not None else "")
    else:
        raise Exception(f"coverage: {coverage}")

    if metric_name == "rouge":
        rouge_metric = evaluate.load("rouge")
        # breakpoint()
        try:
            rouge1_scores_all_instances = rouge_metric.compute(
                references=references, predictions=predictions, use_aggregator=False
            )["rouge1"]
        except Exception as e:
            print(f"\n{e}\n")
            # Seems like if refs is mixed with "null" and other lists, then
            # "null" must come to the first element in the refs, not sure what causes the bug...
            # Workaround: find a "null" in the middle of refs, swap that with the first element
            for i, ref in enumerate(references):
                # Find the first "null" in the middle of refs
                if ref == "null":
                    # Swap places with the first element
                    references[0], references[i] = references[i], references[0]
                    predictions[0], predictions[i] = predictions[i], predictions[0]
                    break
            # also, reload the rouge
            rouge_metric = evaluate.load("rouge")
            # Re-compute the scores
            try:
                rouge1_scores_all_instances = rouge_metric.compute(
                    references=references, predictions=predictions, use_aggregator=False
                )["rouge1"]
            except Exception as e:
                # Debug
                for i in range(len(references)):
                    tmp_preds = predictions[: i + 1]
                    tmp_refs = references[: i + 1]
                    try:
                        tmp_rouge = rouge_metric.compute(
                            references=tmp_refs,
                            predictions=tmp_preds,
                            use_aggregator=False,
                        )["rouge1"]
                    except Exception as e:
                        print(
                            f"compare preds: {str(tmp_preds)} with refs: {str(tmp_refs)} - failed\n"
                        )
                        breakpoint()
                        raise e
                    else:
                        print(
                            f"compare preds: {str(tmp_preds)} with refs: {str(tmp_refs)} - rouge1: {tmp_rouge}\n"
                        )
                raise e

            # raise e
        metric_result = statistics.mean(rouge1_scores_all_instances)
    elif metric_name == "bertscore":
        bertscore = evaluate.load("bertscore")
        try:
            scores = bertscore.compute(
                references=references, predictions=predictions, lang="en"
            )["f1"]
        except Exception as e:
            print(f"\n{e}\n")
            # Seems like if refs is mixed with "null" and other lists, then
            # "null" must come to the first element in the refs, not sure what causes the bug...
            # Workaround: find a "null" in the middle of refs, swap that with the first element
            for i, ref in enumerate(references):
                # Find the first "null" in the middle of refs
                if ref == "null":
                    # Swap places with the first element
                    references[0], references[i] = references[i], references[0]
                    predictions[0], predictions[i] = predictions[i], predictions[0]
                    break
            # also, reload the metric
            bertscore = evaluate.load("bertscore")
            # Re-compute the scores
            try:
                scores = bertscore.compute(
                    references=references, predictions=predictions, lang="en"
                )["f1"]
            except Exception as e:
                # Debug
                for i in range(len(references)):
                    tmp_preds = predictions[: i + 1]
                    tmp_refs = references[: i + 1]
                    try:
                        scores = bertscore.compute(
                            references=tmp_refs, predictions=tmp_preds, lang="en"
                        )["f1"]
                    except Exception as e:
                        print(
                            f"compare preds: {str(tmp_preds)} with refs: {str(tmp_refs)} - failed\n"
                        )
                        breakpoint()
                        raise e
                    else:
                        print(
                            f"compare preds: {str(tmp_preds)} with refs: {str(tmp_refs)} - rouge1: {tmp_rouge}\n"
                        )
                raise e
        metric_result = statistics.mean(scores)

    return metric_result


def evaluate_predicates(
    file_path, evaluate_key, metric_name, yes_count=None, prediction_df=None
):
    """
    Evaluate results at predicate level

    """
    eval_funcs = {
        "slot": eval_slot_predicate_level,
        "answer": eval_answer_predicate_level,
    }
    assert evaluate_key in eval_funcs, (
        f"{evaluate_key} must be one of the `slot`, `answer`"
    )
    if prediction_df is None:
        df = pd.read_csv(file_path)
    else:
        df = prediction_df
    if yes_count is not None:
        df = df[df.yes_count == yes_count]

    df = extract_results_and_add_to_df(file_path, df)

    metric = eval_funcs[evaluate_key](df, metric_name)
    print(file_path, metric)
    # Save the extracted results
    output_dir = Path(file_path).parent
    file_name_stem = Path(file_path).stem.split(".")[0]
    df.to_csv(
        os.path.join(output_dir, f"{file_name_stem}_with_extracted_results.csv"),
        index=False,
    )
    return metric


def extract_results(generation, model_type) -> Dict[str, str]:
    results = {}
    regex_patterns = dict(
        zip(
            ["evidence", "answer", "slot"],
            [r"<think>(.*?)</think>", r"<answer>(.*?)</answer>", r"<slot>(.*?)</slot>"],
        )
    )

    if model_type == "openai":
        generation = eval(generation)
        evidence = generation["reason"]
        answer = generation["answer"]
        slot = generation["slot"]

        for target, (key, pattern) in zip(
            [evidence, answer, slot], regex_patterns.items()
        ):
            matched_strings = re.findall(pattern, target)
            if len(matched_strings) > 0:
                results[key] = matched_strings[0].lower()
            else:
                if key == "answer":
                    if "yes" in target.lower():
                        results[key] = "yes"
                    elif "no" in target.lower():
                        results[key] = "no"
                else:
                    if target != "":
                        results[key] = target

    elif model_type == "hf":
        for key, pattern in regex_patterns.items():
            matched_strings = re.findall(pattern, generation)
            if len(matched_strings) > 0:
                results[key] = matched_strings[0].lower()
            else:
                if key == "answer":
                    if "yes" in generation.lower():
                        results[key] = "yes"
                    elif "no" in generation.lower():
                        results[key] = "no"

    return results


def extract_results_and_add_to_df(file_path, df):
    # Extract `answer` `slot` `evidence` from raw generations and add to the df
    if "prompt_engineering" in file_path:
        model_type = "openai"
        column_name = "response"
    else:
        model_type = "hf"
        column_name = "generated_text"

    df = df.reset_index(drop=True)
    for i, row in df.iterrows():
        extracted_results = extract_results(row[column_name], model_type=model_type)
        for key in ["answer", "slot", "evidence"]:
            # Add the extracted results to the df
            if key in extracted_results:
                df.loc[i, f"predicted_{key}"] = extracted_results[f"{key}"]
            else:
                df.loc[i, f"predicted_{key}"] = None
    return df


def metric_compute(metric_name, references, predictions):
    metric = evaluate.load(metric_name)
    if metric_name == "rouge":
        results = metric.compute(
            references=references, predictions=predictions, use_aggregator=True
        )

    return results


def evaluate_slots(prediction_file: str, labels_file: str, metric_name: str):
    """
    Evaluete slots at ptn level

    """
    labels_df = pd.read_json(labels_file, lines=True)
    ptn_level_results_df = aggregate_predicate_results_to_ptn_results(prediction_file)
    breakpoint()


def aggregate_predicate_results_to_ptn_results(prediction_file: str) -> pd.DataFrame:
    """
    Given a generation file that contains predicate-level results (usually, `testset_results_*?.csv`)
    aggregate to get the ptn-level results, ptns and slots.

    """
    prediction_df = pd.read_csv(prediction_file)
    prediction_df = extract_results_and_add_to_df(prediction_file, prediction_df)
    # breakpoint()

    # Get ptns from predictes
    ptns_predicates_map = {
        1: ["ack-c", "miti"],
        2: ["ack-c", "ano-z"],
        3: ["no-evi"],
        4: ["deny-c", "ano-z"],
        5: ["reverse-c", "trans1"],
        6: ["reverse-c", "trans2"],
        7: ["no-need-address"],
        8: ["y-pro-opposite-z", "y-sup-same-z"],
        9: ["x-pro-z"],
        10: ["x-sup-z"],
    }
    ptn_level_results = []
    for ca_id, indices in prediction_df.groupby("ca_id").groups.items():
        ca_group_df = prediction_df.loc[indices]
        assert len(ca_group_df) == 13, f"{ca_id} has less than 13 predicates"
        yes_predicates = ca_group_df[(ca_group_df.predicted_answer == "yes")][
            "predicate"
        ].tolist()

        # Get ptns
        result_ptns = []
        for ptn, predicates in ptns_predicates_map.items():
            if ptn == 8:
                if set(predicates) & set(yes_predicates):
                    result_ptns.append(ptn)
            else:
                if set(predicates).issubset(set(yes_predicates)):
                    result_ptns.append(ptn)

        # Get slots based on the ptns
        result_slots = defaultdict(list)
        for ptn in result_ptns:
            predicates = ptns_predicates_map[ptn]
            for predicate in predicates:
                row = ca_group_df[ca_group_df.predicate == predicate]
                slot = row.squeeze()["predicted_slot"]

                # breakpoint()

                if isinstance(slot, str):
                    result_slots[ptn].append(slot)

        # Create a new ptn level df
        new_row = {
            key: value
            for key, value in ca_group_df.iloc[0].to_dict().items()
            if key in ["ia_id", "ia", "ca_id", "ca"]
        }
        new_row["predicted_ptns"] = result_ptns
        new_row["predicted_slots"] = result_slots
        ptn_level_results.append(new_row)
    ptn_level_results_df = pd.DataFrame(ptn_level_results)
    return ptn_level_results_df


def evaluate_ptns(prediction_file: str, labels_file: str):
    """
    Evaluate multi-label acc for ptns

    Parameters:
        prediction_file (`str`):
            Predicate-level prediction file located in `testset_outputs` for a model.
        labels_file (`str`):
            JOSONL file that contains all the agg ptns, all_agg_ptns.jsonl
    """

    predicted_ptns_df = aggregate_predicate_results_to_ptn_results(prediction_file)
    labels_df = pd.read_json(labels_file, lines=True)

    # Multi-label acc based on the CALSA+ labels
    predicted_ptns_df = predicted_ptns_df.reset_index(drop=True)
    for i, row in predicted_ptns_df.iterrows():
        predicted_ptns = row["predicted_ptns"]
        # breakpoint()
        label_ptns = labels_df[labels_df.ca_id == row["ca_id"]].squeeze().ptns
        label_slots = labels_df[labels_df.ca_id == row["ca_id"]].squeeze().slots

        if (len(predicted_ptns) == 0) and (len(label_ptns) == 0):
            per_example_acc = 1
        else:
            per_example_acc = len(set(predicted_ptns) & set(label_ptns)) / len(
                set(predicted_ptns) | set(label_ptns)
            )
        predicted_ptns_df.loc[i, "acc"] = per_example_acc

        # Add label ptns and slots to the prediction_df for manual checking purpose
        predicted_ptns_df.loc[i, "label_ptns"] = str(label_ptns)
        predicted_ptns_df.loc[i, "label_slots"] = str(label_slots)

    example_based_acc = sum(predicted_ptns_df.acc.tolist()) / len(predicted_ptns_df)

    print(prediction_file, example_based_acc)

    # Save results
    prediction_file_path = Path(prediction_file)
    output_dir = prediction_file_path.parent
    temperature = prediction_file_path.stem.split("_")[-1]
    # breakpoint()
    predicted_ptns_df.predicted_slots = predicted_ptns_df.apply(
        lambda row: dict(row["predicted_slots"]), axis=1
    )
    predicted_ptns_df.to_csv(
        os.path.join(output_dir, f"predicted_ptns_temperature_{temperature}.csv"),
        index=False,
    )
    return example_based_acc


def read_prediction_file(prediction_file, predicate):
    """
    predicate = all -> return df with all predicates
    predicate = "<specific_predicate>" -> return df with the only the specific_predicate

    """

    all_predicates = [
        "ack-c",
        "ano-z",
        "deny-c",
        "miti",
        "no-evi",
        "no-need-address",
        "reverse-c",
        "trans1",
        "trans2",
        "x-pro-z",
        "x-sup-z",
        "y-pro-opposite-z",
        "y-sup-same-z",
    ]
    df = pd.read_csv(prediction_file)

    if predicate == "all":
        return df
    elif predicate in all_predicates:
        df = df[df.predicate == predicate]
        return df
    else:
        raise Exception(f"predicate: {predicate} is not valid")


def main():
    labels_file = "grpo/data/all_agg_ptns.jsonl"
    # GRPO models
    # common_dir = "grpo/results/deepspeed_zero2"
    # OpenAI models
    common_dir = "prompt_engineering/results/conversational_style"
    # Baseline
    # common_dir = "grpo/results/baseline"
    # SFT
    # common_dir = "sft/results/conversational_style/deepspeed_zero2"

    model_specific_paths = [
        # GRPO
        # "Qwen/Qwen2.5-7B-Instruct/8/fold0/checkpoint-1500/testset_outputs/testset_results_temperature_0.9.csv",
        # "Qwen/Qwen2.5-7B-Instruct/8/fold1/checkpoint-7000/testset_outputs/testset_results_temperature_0.9.csv",
        # "Qwen/Qwen2.5-7B-Instruct/8/fold2/checkpoint-6000/testset_outputs/testset_results_temperature_0.9.csv",
        # "mistralai/Mistral-7B-Instruct-v0.3/8/fold0/checkpoint-8000/testset_outputs/testset_results_temperature_0.9.csv",
        # "mistralai/Mistral-7B-Instruct-v0.3/8/fold1/checkpoint-7000/testset_outputs/testset_results_temperature_0.9.csv",
        # "mistralai/Mistral-7B-Instruct-v0.3/8/fold2/checkpoint-7500/testset_outputs/testset_results_temperature_0.9.csv",
        # "tiiuae/Falcon3-7B-Instruct/8/fold0/checkpoint-3500/testset_outputs/testset_results_temperature_0.9.csv",
        # "tiiuae/Falcon3-7B-Instruct/8/fold1/checkpoint-6000/testset_outputs/testset_results_temperature_0.9.csv",
        # "tiiuae/Falcon3-7B-Instruct/8/fold2/checkpoint-4000/testset_outputs/testset_results_temperature_0.9.csv",
        # OpenAI models
        # "fold0/gpt-4.1-2025-04-14/testset_results.csv",
        # "fold1/gpt-4.1-2025-04-14/testset_results.csv",
        # "fold2/gpt-4.1-2025-04-14/testset_results.csv",
        "fold0/o4-mini-2025-04-16/testset_results.csv",
        "fold1/o4-mini-2025-04-16/testset_results.csv",
        "fold2/o4-mini-2025-04-16/testset_results.csv",
        # Baseline
        # "Qwen/Qwen2.5-7B-Instruct/fold0/testset_results_temperature_0.9.csv",
        # "Qwen/Qwen2.5-7B-Instruct/fold1/testset_results_temperature_0.9.csv",
        # "Qwen/Qwen2.5-7B-Instruct/fold2/testset_results_temperature_0.9.csv",
        # "mistralai/Mistral-7B-Instruct-v0.3/fold0/testset_results_temperature_0.9.csv",
        # "mistralai/Mistral-7B-Instruct-v0.3/fold1/testset_results_temperature_0.9.csv",
        # "mistralai/Mistral-7B-Instruct-v0.3/fold2/testset_results_temperature_0.9.csv",
        # "tiiuae/Falcon3-7B-Instruct/fold0/testset_results_temperature_0.9.csv",
        # "tiiuae/Falcon3-7B-Instruct/fold1/testset_results_temperature_0.9.csv",
        # "tiiuae/Falcon3-7B-Instruct/fold2/testset_results_temperature_0.9.csv",
        # SFT
        # "Qwen/Qwen2.5-7B-Instruct/fold0/checkpoint-30/generation/test_outputs.csv",
        # "Qwen/Qwen2.5-7B-Instruct/fold1/checkpoint-30/generation/test_outputs.csv",
        # "Qwen/Qwen2.5-7B-Instruct/fold2/checkpoint-30/generation/test_outputs.csv",
        # "mistralai/Mistral-7B-Instruct-v0.3/fold0/checkpoint-180/generation/test_outputs.csv",
        # "mistralai/Mistral-7B-Instruct-v0.3/fold1/checkpoint-180/generation/test_outputs.csv",
        # "mistralai/Mistral-7B-Instruct-v0.3/fold2/checkpoint-180/generation/test_outputs.csv",
        # "tiiuae/Falcon3-7B-Instruct/fold0/checkpoint-180/generation/test_outputs.csv",
        # "tiiuae/Falcon3-7B-Instruct/fold1/checkpoint-180/generation/test_outputs.csv",
        # "tiiuae/Falcon3-7B-Instruct/fold2/checkpoint-180/generation/test_outputs.csv",
    ]

    print("ptns - multi-label acc")
    results_all_folds = []
    for model_specific_path in model_specific_paths:
        prediction_file = f"{common_dir}/{model_specific_path}"
        results_all_folds.append(evaluate_ptns(prediction_file, labels_file))

    # Workaround for calculating results for a single checkpoint
    # results_all_folds = results_all_folds * 3
    std = statistics.stdev(results_all_folds)
    mean = statistics.mean(results_all_folds)
    print(f"All: {results_all_folds}")
    print(f"Avg.: {mean}")
    print(f"Std.: {std}")
    print("\n\n")

    metrics_mapping = {
        "answer": ["macro_f1"],
        "slot": ["rouge_all", "bertscore_all", "rouge_r", "bertscore_r"],
    }

    for evaluation_key, metric_names in metrics_mapping.items():
        for metric_name in metric_names:
            print(f"{evaluation_key} - {metric_name}")
            results_all_folds = []
            for model_specific_path in model_specific_paths:
                prediction_file = f"{common_dir}/{model_specific_path}"

                # Can also pass prediction_df directly for evaluation
                predicate = "x-sup-z"
                prediction_df = read_prediction_file(
                    prediction_file, predicate=predicate
                )
                # breakpoint()
                print(f"predicate: {predicate}")

                results_all_folds.append(
                    evaluate_predicates(
                        file_path=prediction_file,
                        evaluate_key=evaluation_key,
                        metric_name=metric_name,
                        prediction_df=prediction_df,
                    )
                )

            # Workaround for calculating results for a single checkpoint
            # results_all_folds = results_all_folds * 3
            std = statistics.stdev(results_all_folds)
            mean = statistics.mean(results_all_folds)
            print(f"All: {results_all_folds}")
            print(f"Avg.: {mean}")
            print(f"Std.: {std}")
            print("\n\n")


if __name__ == "__main__":
    main()
