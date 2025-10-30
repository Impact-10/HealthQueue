import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/flan-t5-small"

prompt = "What should I do if I have a high fever and sore throat?"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)



inputs = tokenizer(prompt, return_tensors="pt").to(device)
print("Input tensor shape:", inputs["input_ids"].shape)
output = model.generate(
	**inputs,
	max_length=50,
	do_sample=True,
	temperature=0.9,
	top_p=0.95,
	num_return_sequences=1
)
raw_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("\n=== BioGPT Output ===\n")
print(raw_text)
