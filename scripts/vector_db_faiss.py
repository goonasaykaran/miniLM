import faiss
import numpy as np
import json

# ✅ Load embeddings
embeddings = np.load("data/embeddings.npy")

# ✅ Load text chunks
with open("data/embeddings.json", "r") as f:
    text_chunks = json.load(f)["texts"]

# ✅ Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ✅ Save FAISS index
faiss.write_index(index, "vector_db/faiss_index.bin")

# ✅ Save text chunks separately
with open("vector_db/text_chunks.json", "w") as f:
    json.dump(text_chunks, f, indent=4)

print(f"✅ Stored {len(text_chunks)} embeddings in FAISS!")
