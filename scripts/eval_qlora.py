# scripts/eval_qlora.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path

# === Paths
ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_DIR = (ROOT / "models" / "base" / "Qwen2.5-0.5B").as_posix()
QLORA_ADAPTER_DIR = (ROOT / "models" / "adapters" / "qwen-0.5b-qlora").as_posix()

# === Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True
)

# === Load base model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True,
    quantization_config=bnb_config,
    device_map="auto"
)

# === Load QLoRA adapter
model = PeftModel.from_pretrained(base_model, QLORA_ADAPTER_DIR)
model.eval()

# === Generation
def generate_response(prompt, max_new_tokens=128):
    input_text = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Example prompts
examples = [
    "Translate to French: Hello, how are you?",
    "Summarize: The stock market fluctuated wildly today after economic news.",
    "Write a short story about a dog who learns to fly.",
    "Who is the shortest if Mary > John > Bob?",
    "Is it okay to steal if you're hungry?",
    "How can I improve my resume?",
    "Oh great. Try that again, genius."
]

for i, prompt in enumerate(examples):
    print(f"\n🧪 Example {i+1}: {prompt}")
    output = generate_response(prompt)
    print("📤 QLoRA Output:\n", output)
