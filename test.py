import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import os
import re

# ===============================
# Custom Model Definition
# ===============================
class ParamExtractionModel(nn.Module):
    def __init__(self, base_model, num_params):
        super().__init__()
        self.base_model = base_model
        embedding_dim = base_model.get_sentence_embedding_dimension()
        self.param_head = nn.Linear(embedding_dim, num_params + 1)  # +1 for non-param tokens

    def forward(self, inputs, return_sentence_embedding=False):
        outputs = self.base_model(inputs)
        sentence_embedding = outputs["sentence_embedding"]
        token_embeddings = outputs["token_embeddings"]
        param_logits = self.param_head(token_embeddings)
        if return_sentence_embedding:
            return {"sentence_embedding": sentence_embedding, "param_logits": param_logits}
        return {"param_logits": param_logits}

# ===============================
# Load JSONL or fallback data
# ===============================
def load_data(filepath="function_training_data.jsonl"):
    if not os.path.exists(filepath):
        print(f"File {filepath} not found. Using default data.")
        return [
            {"prompt": "open firefox and go to google.com",
             "context": "Opens Firefox and navigates to specified website.",
             "function": "open_firefox(website: str)",
             "args": {"website": "google.com"},
             "param_annotations": {"website": ["google.com"]}},
            {"prompt": "send an email to John with subject Hello",
             "context": "Sends an email to the recipient with subject and body.",
             "function": "send_email(to: str, subject: str, body: str)",
             "args": {"to": "John", "subject": "Hello", "body": ""},
             "param_annotations": {"to": ["John"], "subject": ["Hello"], "body": [""]}},
        ]
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

# ===============================
# Load param_to_idx mapping
# ===============================
def load_param_to_idx(filepath="param_to_idx.json"):
    if not os.path.exists(filepath):
        print(f"Warning: '{filepath}' not found. Cannot map parameters by name.")
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# ===============================
# Extract parameters using model predictions
# ===============================
def extract_parameters(model, tokenizer, prompt, idx_to_param, device="cpu"):
    model.eval()
    model.to(device)
    inputs = model.base_model.tokenize([prompt])
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(inputs, return_sentence_embedding=True)
    param_logits = outputs["param_logits"][0]
    pred_indices = torch.argmax(F.softmax(param_logits, dim=-1), dim=-1).cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

    params = {}
    current_param = None
    current_tokens = []

    for tok, idx in zip(tokens, pred_indices):
        if idx == 0:
            if current_param and current_tokens:
                val = tokenizer.convert_tokens_to_string(current_tokens).replace("##", "").strip()
                params.setdefault(current_param, []).append(val)
                current_param, current_tokens = None, []
            continue
        param_name = idx_to_param.get(str(idx), f"param{idx}")
        if param_name != current_param:
            if current_param and current_tokens:
                val = tokenizer.convert_tokens_to_string(current_tokens).replace("##", "").strip()
                params.setdefault(current_param, []).append(val)
            current_param = param_name
            current_tokens = [tok]
        else:
            current_tokens.append(tok)
    if current_param and current_tokens:
        val = tokenizer.convert_tokens_to_string(current_tokens).replace("##", "").strip()
        params.setdefault(current_param, []).append(val)

    # Join multi-token values
    for k in params:
        params[k] = " ".join(params[k])
    return params

# ===============================
# Fallback regex-based extraction for unseen dynamic parameters
# ===============================
def extract_dynamic_params(prompt, function_name):
    params = {}
    if function_name in ["open_firefox(website: str)", "open_chrome(website: str)"]:
        match = re.search(r"to (\S+)", prompt)
        if match:
            params["website"] = match.group(1)
    return params

# ===============================
# Main Inference
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load training data
data = load_data()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Load checkpoint
checkpoint = torch.load("model.pt", map_location=device)
num_params = checkpoint['param_head.weight'].shape[0] - 1  # subtract 1 for non-param token

# Initialize model
base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = ParamExtractionModel(base_model, num_params=num_params)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# Load param_to_idx mapping
param_to_idx = load_param_to_idx()
if param_to_idx is not None:
    idx_to_param = {v: k for k, v in param_to_idx.items()}
else:
    idx_to_param = {i+1: f"param{i+1}" for i in range(num_params)}

# Encode context embeddings once
contexts = [d["context"] for d in data]
context_embeddings = model.base_model.encode(contexts, convert_to_tensor=True, batch_size=2).to(device)

# Example prompt
test_prompt = "open notepad and write Hello World in it"

# Extract parameters from model
model_params = extract_parameters(model, tokenizer, test_prompt, idx_to_param, device=device)

# Match context
prompt_embedding = model.base_model.encode([test_prompt], convert_to_tensor=True).to(device)
similarities = F.cosine_similarity(prompt_embedding, context_embeddings)
best_idx = similarities.argmax().item()
best_context = contexts[best_idx]
best_function = data[best_idx]["function"]
best_args = data[best_idx]["args"].copy()

# Extract dynamic params (regex) and merge
dynamic_params = extract_dynamic_params(test_prompt, best_function)
final_args = best_args.copy()
final_args.update(model_params)
final_args.update(dynamic_params)  # dynamic extraction overrides if present

# Output results
print("Prompt:", test_prompt)
print("Best Match Context:", best_context)
print("Best Match Function:", best_function)
print("Extracted Args:", final_args)
print("Similarity Score:", float(similarities[best_idx]))
