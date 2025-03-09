import torch
from sentence_transformers import SentenceTransformer, losses, InputExample, LoggingHandler
from torch.utils.data import DataLoader
import json

# ✅ Load base model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# ✅ Load training data
train_samples = []
with open("data/training_data.json", "r") as f:
    data = json.load(f)
    for sample in data:
        train_samples.append(InputExample(texts=[sample["text1"], sample["text2"]], label=float(sample["label"])))

# ✅ Convert to DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)

# ✅ Use Contrastive Loss
train_loss = losses.CosineSimilarityLoss(model)

# ✅ Train Model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

# ✅ Save Fine-Tuned Model
model.save("hf_models/fine_tuned_minilm")
print("✅ Model fine-tuned and saved!")
