Run Training & API

 1️⃣ Install Dependencies
 pip install -r requirements.txt

2️⃣ Choose Your Model (Edit config.json)
To use MiniLM:
    "active_model": "minilm"
To use LLAMA:
    "active_model": "llama"

3️⃣ Generate Embeddings
python scripts/generate_embeddings.py

4️⃣ Store Embeddings in FAISS
python scripts/vector_db_faiss.py

5️⃣ Fine-tune MiniLM:
python scripts/fine_tune_embeddings.py

6️⃣  Fine-tune LLAMA:
python scripts/fine_tune_llama.py

7️⃣ Start Flask API with fine-tuned model:
python api/flask_search_api.py

8️⃣ Access Web UI:
Visit http://127.0.0.1:5000 in your browser.