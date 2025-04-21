from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B"
save_directory = "./Qwen2.5-0.5B"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save them locally
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
#