import torch
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
import json
from datasets import Dataset

# ✅ Load LLAMA model & tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue

model = AutoModel.from_pretrained(model_name)

# ✅ Prepare Data
train_data = []
with open("data/training_data.json", "r") as f:
    data = json.load(f)
    for sample in data:
        train_data.append({"input_ids": tokenizer(sample["text1"], truncation=True, padding="max_length", max_length=512)["input_ids"],
                           "attention_mask": tokenizer(sample["text1"], truncation=True, padding="max_length", max_length=512)["attention_mask"],
                           "labels": tokenizer(sample["text2"], truncation=True, padding="max_length", max_length=512)["input_ids"]})

dataset = Dataset.from_dict(train_data)

# ✅ Training Arguments
training_args = TrainingArguments(
    output_dir="hf_models/fine_tuned_llama",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

# ✅ Train Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

# ✅ Save Fine-Tuned Model
model.save_pretrained("hf_models/fine_tuned_llama")
print("✅ LLAMA model fine-tuned and saved!")
