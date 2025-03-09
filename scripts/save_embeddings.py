import json
import numpy as np
import pandas as pd

# Load extracted text chunks
from scripts.json_to_chunks import extract_chunks

with open("data/sample.json", "r") as file:
    json_data = json.load(file)

text_chunks = extract_chunks(json_data)

# Load embeddings
embeddings = np.load("data/embeddings.npy")

# Save as JSON
embedding_data = {"texts": text_chunks, "embeddings": embeddings.tolist()}
with open("data/embeddings.json", "w") as f:
    json.dump(embedding_data, f, indent=4)

print("✅ Embeddings saved in 'data/embeddings.json'")

# Save as CSV
df = pd.DataFrame(embeddings)
df.insert(0, "text", text_chunks)  # Attach text
df.to_csv("data/embeddings.csv", index=False)

print("✅ Embeddings saved in 'data/embeddings.csv'")
