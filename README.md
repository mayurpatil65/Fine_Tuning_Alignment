
# SFT and Policy Alignment on Local Models

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

> This is a fully local POC developed for research and internal use.
 
## Key Features

### SFT with LoRA 
Dataset Preparation:
Download the Raw Dataset
```data/get_dataset.py```

Process dataset to Instruction Format:
```utils/data_utils.py```

Model Preparation:
Download Base Model Locally
```models/base/get_model.py```

Fine-Tuning with LoRA:
```scripts/train_lora_sft.py```

Fine-tuned adapters will be saved in:
```models/adapters/qwen-0.5b-lora/```

Evaluation:
Evaluate the Base Model
```scripts/eval_base.py```
 
Evaluate the Fine-Tuned (LoRA) Model
```scripts/eval_lora.py```

### SFT with QLoRA 
Similar structure to LoRA

### SFT Full Fine Tune
#### Unresonable to execute locally

## Dependencies 

transformers

datasets

peft

torch

accelerate

tensorboard

 

## Final Notes
- Fully local: base model, adapters, etc

- SFT with LoRA, QLoRA, Full fine-tune

- Policy Alignment with DPO, RLFH (PPO), RLAIF, GRPO

- Modular: Easily extend with new techniques


## Roadmap
 1. Future stuff


