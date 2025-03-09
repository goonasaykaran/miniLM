import faiss
import numpy as np
import json

# ✅ Load FAISS index
index = faiss.read_index("vector_db/faiss_index.bin")

# ✅ Load text chunks
with open("vector_db/text_chunks.json", "r") as f:
    text_chunks = json.load(f)

# ✅ Extract stored embeddings using reconstruct()
num_vectors = index.ntotal  # Get the number of stored embeddings
dimension = index.d  # Get the embedding dimension

# Create an empty NumPy array to store embeddings
stored_embeddings = np.zeros((num_vectors, dimension), dtype=np.float32)

# Retrieve all stored vectors
for i in range(num_vectors):
    stored_embeddings[i] = index.reconstruct(i)  # Extract each vector

# ✅ Convert embeddings to list for JSON storage
embedding_data = {
    "texts": text_chunks,
    "embeddings": stored_embeddings.tolist()
}

# ✅ Save extracted embeddings to JSON
with open("data/extracted_embeddings.json", "w") as f:
    json.dump(embedding_data, f, indent=4)

print(f"✅ Extracted {num_vectors} embeddings from FAISS and saved to 'data/extracted_embeddings.json'!")
