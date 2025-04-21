from pathlib import Path
import json
import os
#
def preprocess_oasst1_to_sft(
):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    raw_path = PROJECT_ROOT / "data/raw/oasst1.json"
    out_path = PROJECT_ROOT / "data/processed/sft_data.json"
    raw_path = Path(raw_path)
    out_path = Path(out_path)

    parent_map = {}
    formatted = []

    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            parent_map[ex["message_id"]] = ex

    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if ex["role"] != "assistant" or not ex.get("parent_id"):
                continue
            parent = parent_map.get(ex["parent_id"])
            if not parent: continue
            formatted.append({
                "instruction": parent["text"],
                "output": ex["text"]
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in formatted:
            f.write(json.dumps(ex) + "\n")

    print(f"✅ Saved {len(formatted)} instruction pairs → {out_path}")

preprocess_oasst1_to_sft()