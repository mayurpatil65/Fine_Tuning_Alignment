
from datasets import load_dataset
import json
import os

output_path = "data/raw/oasst1.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

dataset = load_dataset("OpenAssistant/oasst1", split="train")

with open(output_path, "w", encoding="utf-8") as f:
    for ex in dataset:
        f.write(json.dumps(ex) + "\n")

print(f"Saved {len(dataset)} rows to {output_path}")
#