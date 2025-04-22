# scripts/eval_full.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# === Enable full CUDA error traceback if needed ===
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# === Paths
ROOT = Path(__file__).resolve().parents[1]
FULL_MODEL_DIR = (ROOT / "models" / "adapters" / "qwen-0.5b-full").as_posix()

# === Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    FULL_MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True
)

# === Load full fine-tuned model
model = AutoModelForCausalLM.from_pretrained(
    FULL_MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# === Disable unsupported sliding window attention if present
if hasattr(model.config, "sliding_window"):
    model.config.sliding_window = None

# === Generation
def generate_response(prompt, max_new_tokens=128):
    input_text = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Check logits before generating
    with torch.no_grad():
        logits = model(**inputs).logits
        if torch.isnan(logits).any():
            raise RuntimeError("❌ NaNs detected in model logits")
        probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        if torch.sum(probs, dim=-1).eq(0).any():
            raise RuntimeError("❌ Invalid probability distribution detected")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Example prompts
examples = [
    "Translate to French: Hello, how are you?",
    "Summarize: The stock market fluctuated wildly today after economic news.",
    "Write a short story about a dog who learns to fly.",
    "How do I become a better public speaker?",
    "What are the benefits of regular exercise?",
    "Is it okay to steal if you're hungry?",
    "Oh great. Try that again, genius."
]

for i, prompt in enumerate(examples):
    print(f"\n🧪 Example {i+1}: {prompt}")
    try:
        output = generate_response(prompt)
        print("📤 Full Fine-tuned Output:\n", output)
    except Exception as e:
        print("❌ Error during generation:", e)
