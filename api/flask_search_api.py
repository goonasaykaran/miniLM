from flask import Flask, request, jsonify, render_template
import faiss
import numpy as np
import json
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

app = Flask(__name__, template_folder="templates", static_folder="static")

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

# ✅ Load FAISS index
index = faiss.read_index("vector_db/faiss_index.bin")

# ✅ Load text chunks
with open("vector_db/text_chunks.json", "r") as f:
    text_chunks = json.load(f)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    text = request.form["text"]
    top_n = int(request.form.get("top_n", 3))

    # ✅ Generate embedding
    query_embedding = np.array([get_embedding(text)])

    # ✅ Perform FAISS search
    distances, indices = index.search(query_embedding, top_n)

    results = [{"text": text_chunks[i], "score": round(float(distances[0][j]), 4)} for j, i in enumerate(indices[0])]

    # ✅ Create hierarchical structure for D3.js
    dependency_tree = {
        "name": text,
        "children": [{"name": result["text"], "score": result["score"]} for result in results]
    }

    return render_template("index.html", query=text, results=results, tree_data=json.dumps(dependency_tree))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
from flask import Flask, request, jsonify, render_template
import faiss
import numpy as np
import json
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

app = Flask(__name__, template_folder="templates", static_folder="static")

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

# ✅ Load FAISS index
index = faiss.read_index("vector_db/faiss_index.bin")

# ✅ Load text chunks
with open("vector_db/text_chunks.json", "r") as f:
    text_chunks = json.load(f)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    text = request.form["text"]
    top_n = int(request.form.get("top_n", 3))

    # ✅ Generate embedding
    query_embedding = np.array([get_embedding(text)])

    # ✅ Perform FAISS search
    distances, indices = index.search(query_embedding, top_n)

    results = [{"text": text_chunks[i], "score": round(float(distances[0][j]), 4)} for j, i in enumerate(indices[0])]

    # ✅ Create hierarchical structure for D3.js
    dependency_tree = {
        "name": text,
        "children": [{"name": result["text"], "score": result["score"]} for result in results]
    }

    return render_template("index.html", query=text, results=results, tree_data=json.dumps(dependency_tree))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
from flask import Flask, request, jsonify, render_template
import faiss
import numpy as np
import json
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

app = Flask(__name__, template_folder="templates", static_folder="static")

# ✅ Load config file
with open("config.json", "r") as f:
    config = json.load(f)

active_model = config["active_model"]  # Select model
model_name = config["models"][active_model]

print(f"✅ Using model: {model_name}")

# ✅ Load model based on config
# Load Fine-Tuned Model
if active_model == "llama":
    model = AutoModel.from_pretrained("hf_models/fine_tuned_llama")
else:
    model = SentenceTransformer("hf_models/fine_tuned_minilm")


# ✅ Load FAISS index
index = faiss.read_index("vector_db/faiss_index.bin")

# ✅ Load text chunks
with open("vector_db/text_chunks.json", "r") as f:
    text_chunks = json.load(f)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    text = request.form["text"]
    top_n = int(request.form.get("top_n", 3))

    # ✅ Generate embedding
    query_embedding = np.array([get_embedding(text)])

    # ✅ Perform FAISS search
    distances, indices = index.search(query_embedding, top_n)

    results = [{"text": text_chunks[i], "score": round(float(distances[0][j]), 4)} for j, i in enumerate(indices[0])]

    # ✅ Create hierarchical structure for D3.js
    dependency_tree = {
        "name": text,
        "children": [{"name": result["text"], "score": result["score"]} for result in results]
    }

    return render_template("index.html", query=text, results=results, tree_data=json.dumps(dependency_tree))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
from flask import Flask, request, jsonify, render_template
import faiss
import numpy as np
import json
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

app = Flask(__name__, template_folder="templates", static_folder="static")

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

# ✅ Load FAISS index
index = faiss.read_index("vector_db/faiss_index.bin")

# ✅ Load text chunks
with open("vector_db/text_chunks.json", "r") as f:
    text_chunks = json.load(f)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    text = request.form["text"]
    top_n = int(request.form.get("top_n", 3))

    # ✅ Generate embedding
    query_embedding = np.array([get_embedding(text)])

    # ✅ Perform FAISS search
    distances, indices = index.search(query_embedding, top_n)

    results = [{"text": text_chunks[i], "score": round(float(distances[0][j]), 4)} for j, i in enumerate(indices[0])]

    # ✅ Create hierarchical structure for D3.js
    dependency_tree = {
        "name": text,
        "children": [{"name": result["text"], "score": result["score"]} for result in results]
    }

    return render_template("index.html", query=text, results=results, tree_data=json.dumps(dependency_tree))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
