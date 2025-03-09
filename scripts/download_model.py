from transformers import AutoModel, AutoTokenizer

# Define the model name
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Download and cache the model locally
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./hf_models")
model = AutoModel.from_pretrained(model_name, cache_dir="./hf_models")

print("Model downloaded and stored in './hf_models' directory.")
