# CALSA+ dataset and other additional materials for the paper: Identification of Multiple Logical Interpretations in Counter-Arguments

`dataset` folder contains the CALSA+ dataset and the related files

- `processed_merged_results_readable.csv`: Raw annotation results of three annotators at the predicate-level
- `all.jsonl`: processed predicate-level results, the answer is determined by taking as many `YES` as possible (i.e., answer = `YES` if any of the annotators select `YES`; otherwise, `NO`)
- `all_agg_ptns.jsonl`: The CALSA+ dataset where each CA has multiple logical interpretations obtained by aggregating the predicate-level results in `all.jsonl`
    - The ptn numbers are consistent with the patterns of logical structure defined in original CALSA paper: https://aclanthology.org/2024.findings-emnlp.661/

`grpo` folder contains all the data and scripts for RLVR experiments 

- For training models with RLVR, run any of the scripts in `grpo/src/scripts/train/`, each one is corresponded to a base model 
    - e.g., `cd <where-you-clone-this-repo> && source grpo/src/scripts/train/qwen25-7b-instruct_deepspeed_zero2.sh`
- For runing inferences, run any of the scripts in `grpo/src/scripts/inference/`, each one is corresponded to a base model 
    - e.g., `cd <where-you-clone-this-repo> && source grpo/src/scripts/inference/qwen25-7b-instruct_deepspeed_zero2.sh`
- For running baseline experiments to compare with RLVR, run the corresponding script in `grpo/src/scripts/inference/baselines/` 

`sft` folder contains all the data and scripts for SFT experiments 

- For training models with SFT, run any of the scripts in `sft/src/scripts/train/`, each one is corresponded to a base model 
    - e.g., `cd <where-you-clone-this-repo> && source sft/src/scripts/train/qwen25-7b-instruct_deepspeed_zero2.sh`
- For runing inferences, run any of the scripts in `sft/src/scripts/inference/`, each one is corresponded to a base model 
    - e.g., `cd <where-you-clone-this-repo> && source sft/src/scripts/inference/qwen25-7b-instruct_deepspeed_zero2.sh`

`prompt_enginneering` folder contains all the data and scripts for prompting OpenAI models

- Usage: `python prompt_engineering/src/gpts.py`

For evaluting any of the above experiments, run `python evaluate_results.py` with the corresponding file path where you saved the results
