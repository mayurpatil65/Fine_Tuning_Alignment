# scripts/train_full_sft.py

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from pathlib import Path
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# === Local paths
ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_DIR = ROOT / "models" / "base" / "Qwen2.5-0.5B"
DATA_PATH = ROOT / "data" / "processed" / "sft_data.json"
OUTPUT_DIR = ROOT / "models" / "adapters" / "qwen-0.5b-full"
LOG_DIR = ROOT / "logs" / "tensorboard"

# === Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True
)

# === Load full model (no LoRA/PEFT)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# === Tie weights (Qwen requires it)
if hasattr(model, "tie_weights"):
    model.tie_weights()

# === Load dataset
def load_local_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return Dataset.from_list(rows)

dataset = load_local_dataset(DATA_PATH)

def tokenize(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=256)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(tokenize, remove_columns=["instruction", "output"])
dataset = dataset.select(range(500))  # small subset for testing

# === Training setup
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=1,  # reduce for 8GB GPU
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    logging_dir=str(LOG_DIR),
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="tensorboard",
    learning_rate=2e-4,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    lr_scheduler_type="cosine",
    warmup_steps=50,
    gradient_checkpointing=True,
    dataloader_num_workers=0
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# === Save results
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Full fine-tuning complete. Model saved to: {OUTPUT_DIR}")
