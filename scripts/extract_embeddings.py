import json
import numpy as np

# Load embeddings
embeddings = np.load("data/embeddings.npy")

# Load text chunks
with open("vector_db/text_chunks.json", "r") as f:
    text_chunks = json.load(f)

# Save as JSON
embedding_data = {"texts": text_chunks, "embeddings": embeddings.tolist()}
with open("data/extracted_embeddings.json", "w") as f:
    json.dump(embedding_data, f, indent=4)

print("✅ Extracted embeddings saved in 'data/extracted_embeddings.json'")
