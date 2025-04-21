This is prototype to do the following
1. SFT - LoRA, QLoRA, Full fine tune
2. Policy Alignment - DPO, RLFH (PPO), RLAIF, GRPO,

Uses the following 
1. Local training only
2. No Hugging Face uploads - everything is local
3. Base Model Qwen2.5-0.5B or Qwen2.5-1.5B
4. Alignment Method LoRA SFT, then later DPO
5. Dataset for SFT - OpenAssistant/oasst1
6. GPU RTX 2070 (8GB)

Need the following packages

1. transformers
2. datasets
3. peft
4. torch
5. accelerate
6. tensorboard

# Part 1 SFT LoRA
1. Dataset Preparation:
Download the Raw Dataset
```data/get_dataset.py```
2. Process dataset to Instruction Format:
```utils/data_utils.py```
3. Model Preparation:
Download Base Model Locally
```models/base/get_model.py```
4. Fine-Tuning with LoRA:
```scripts/train_lora_sft.py```
5. Fine-tuned adapters will be saved in:
```models/adapters/qwen-0.5b-lora/```
6. Evaluation:
Evaluate the Base Model
```scripts/eval_base.py```
7. Evaluate the Fine-Tuned (LoRA) Model
```scripts/eval_lora.py```


