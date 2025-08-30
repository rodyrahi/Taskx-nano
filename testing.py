import torch
import json
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from train import ParamExtractionModel, extract_parameters  # import your model class and extract_parameters

# ===============================
# Settings
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
param_model_path = "param_model.pt"
param_to_idx_path = "param_to_idx.json"
functions_jsonl_path = "function_training_data.jsonl"  # your JSONL of functions

# ===============================
# Load trained parameter extraction model
# ===============================
base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

with open(param_to_idx_path, "r") as f:
    param_to_idx = json.load(f)
num_params = len(param_to_idx)

param_model = ParamExtractionModel(base_model, num_params)
param_model.load_state_dict(torch.load(param_model_path, map_location=device))
param_model.to(device)
param_model.eval()  # IMPORTANT: ensures no training

# ===============================
# Load functions from JSONL
# ===============================
functions = []
func_texts = []

with open(functions_jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        func = json.loads(line)
        functions.append(func)
        func_texts.append(f"{func['function_signature']} | {func['context']} | {func['args']}")

# ===============================
# Build FAISS index
# ===============================
func_embeddings = base_model.encode(func_texts, convert_to_numpy=True)
faiss.normalize_L2(func_embeddings)

dim = func_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(func_embeddings)

# ===============================
# Predict best function + extract parameters
# ===============================
def predict_best_function(prompt, top_k=1):
    # Encode prompt
    prompt_emb = base_model.encode([prompt], convert_to_numpy=True)
    faiss.normalize_L2(prompt_emb)

    # Search in FAISS
    D, I = index.search(prompt_emb, k=top_k)
    best_idx = I[0][0]
    best_func = functions[best_idx]

    # Extract parameters using the trained model
    extracted_params = extract_parameters(param_model, tokenizer, prompt, param_to_idx, device=device)
    return best_func, extracted_params

# ===============================
# Example Usage
# ===============================
if __name__ == "__main__":
    test_prompts = [
        "open 'kamingo.in' in chrome",
        "set a reminder for my meeting at 3 PM tomorrow",
        "send an email to Alice with subject Meeting and body Please review the report",
        "play the song 'Levitating' by Dua Lipa"
    ]

    for prompt in test_prompts:
        best_func, params = predict_best_function(prompt)
        print("\nPrompt:", prompt)
        print("Best Matching Function:", best_func['function_signature'])
        print("Extracted Parameters:", params)
