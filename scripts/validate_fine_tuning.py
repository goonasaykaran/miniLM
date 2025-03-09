import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load test dataset
with open("data/test_sentences.json", "r") as file:
    test_data = json.load(file)

similar_pairs = test_data["similar_pairs"]
dissimilar_pairs = test_data["dissimilar_pairs"]

# Load models before and after fine-tuning
pretrained_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
fine_tuned_model = SentenceTransformer("hf_models/fine_tuned_model")

def compute_similarity(model, sentence_pairs):
    """Compute cosine similarity between sentence pairs."""
    similarities = []
    for sent1, sent2 in sentence_pairs:
        emb1 = model.encode([sent1])
        emb2 = model.encode([sent2])
        score = cosine_similarity(emb1, emb2)[0][0]
        similarities.append(score)
    return np.mean(similarities), similarities

# Compute similarity before fine-tuning
pre_similar_avg, pre_similar_scores = compute_similarity(pretrained_model, similar_pairs)
pre_dissimilar_avg, pre_dissimilar_scores = compute_similarity(pretrained_model, dissimilar_pairs)

# Compute similarity after fine-tuning
post_similar_avg, post_similar_scores = compute_similarity(fine_tuned_model, similar_pairs)
post_dissimilar_avg, post_dissimilar_scores = compute_similarity(fine_tuned_model, dissimilar_pairs)

# Print evaluation results
print("\n🔹 **Model Evaluation Results** 🔹\n")
print(f"✅ Similar Sentences Before Fine-Tuning: {pre_similar_avg:.4f}")
print(f"✅ Similar Sentences After Fine-Tuning: {post_similar_avg:.4f}\n")

print(f"❌ Dissimilar Sentences Before Fine-Tuning: {pre_dissimilar_avg:.4f}")
print(f"❌ Dissimilar Sentences After Fine-Tuning: {post_dissimilar_avg:.4f}\n")

improvement = (post_similar_avg - pre_similar_avg) - (pre_dissimilar_avg - post_dissimilar_avg)
print(f"🚀 **Fine-Tuning Impact Score: {improvement:.4f}** (Higher is better)\n")

# ✅ Convert NumPy float32 to standard Python float
eval_results = {
    "before_fine_tuning": {
        "similar_avg": float(pre_similar_avg),  # Convert to standard float
        "dissimilar_avg": float(pre_dissimilar_avg)
    },
    "after_fine_tuning": {
        "similar_avg": float(post_similar_avg),
        "dissimilar_avg": float(post_dissimilar_avg)
    },
    "impact_score": float(improvement)
}


with open("data/fine_tuning_evaluation.json", "w") as f:
    json.dump(eval_results, f, indent=4)

print("📊 Evaluation results saved to 'data/fine_tuning_evaluation.json'!")
