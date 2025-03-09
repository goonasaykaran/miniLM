import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from json_to_chunks import extract_chunks

# ✅ Load config file
with open("config.json", "r") as f:
    config = json.load(f)

active_model = config["active_model"]  # Select model
model_name = config["models"][active_model]

print(f"✅ Using model: {model_name}")

# ✅ Load model based on config
if active_model == "llama":
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

else:
    model = SentenceTransformer(model_name)

    def get_embedding(text):
        return model.encode(text)

# ✅ Load JSON data
with open("data/sample.json", "r") as file:
    json_data = json.load(file)

# ✅ Extract text chunks
text_chunks = extract_chunks(json_data)

# ✅ Generate embeddings
embeddings = np.array([get_embedding(text) for text in text_chunks])

# ✅ Save embeddings
np.save("data/embeddings.npy", embeddings)
with open("data/embeddings.json", "w") as f:
    json.dump({"texts": text_chunks, "embeddings": embeddings.tolist()}, f, indent=4)

print(f"✅ Generated {len(text_chunks)} embeddings using {active_model.upper()}!")
