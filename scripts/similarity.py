import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load saved embeddings and texts
with open("data/embeddings.json", "r") as file:
    embedding_data = json.load(file)

text_chunks = embedding_data["texts"]
embeddings = np.array(embedding_data["embeddings"])

# Function to find similar chunks
def find_similar(text, model, top_n=3):
    query_embedding = model.encode([text])
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Get top N similar chunks
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [(text_chunks[i], similarities[i]) for i in top_indices]

# Example Search
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    model_path = "./hf_models/sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_path)

    search_text = "Gaming Laptop with NVIDIA GPU"
    results = find_similar(search_text, model)

    print(f"\n🔍 Similar Chunks to '{search_text}':")
    for text, score in results:
        print(f"  🔹 {text} (Score: {score:.4f})")
