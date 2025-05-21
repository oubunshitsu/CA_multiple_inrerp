# CALSA+ dataset and other additional materials for the paper: Identification of Multiple Logical Interpretations in Counter-Arguments

## The repo current only contains the CALSA+ dataset, the scripts used for training the models will be added soon.

## Details of each file

- `processed_merged_results_readable.csv`: Raw annotation results of three annotators at the predicate-level
- `all.jsonl`: processed predicate-level results, the answer is determined by taking as many `YES` as possible (i.e., answer = `YES` if any of the annotators select `YES`; otherwise, `NO`)
- `all_agg_ptns.jsonl`: The CALSA+ dataset where each CA has multiple logical interpretations obtained by aggregating the predicate-level results in `all.jsonl`
    - The ptn numbers are consistent with the patterns of logical structure defined in original CALSA paper: https://aclanthology.org/2024.findings-emnlp.661/


