import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from transformers import AutoTokenizer

# ===============================
# Custom Parameter Extraction Model
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
# Contrastive Loss
# ===============================
def contrastive_loss(prompt_embs, func_embs, temperature=0.07):
    sim_matrix = torch.matmul(prompt_embs, func_embs.T) / temperature
    labels = torch.arange(len(prompt_embs)).to(prompt_embs.device)
    loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
    return loss / 2

# ===============================
# Load JSONL Data
# ===============================
def load_data_from_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    return data

# ===============================
# Prepare Training Examples
# ===============================
def prepare_training_examples(data):
    param_to_idx = {}
    idx = 1
    for d in data:
        for param in d["param_annotations"]:
            if param not in param_to_idx:
                param_to_idx[param] = idx
                idx += 1
    with open("param_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(param_to_idx, f)

    train_examples = [
        InputExample(texts=[d["prompt"], d["context"]], guid=d["param_annotations"])
        for d in data
    ]
    return train_examples, param_to_idx

# ===============================
# Collate Function
# ===============================
def custom_collate_fn(batch, tokenizer, param_to_idx):
    prompts = [ex.texts[0] for ex in batch]
    contexts = [ex.texts[1] for ex in batch]
    param_labels = []

    for ex in batch:
        annotations = ex.guid
        tokenized = tokenizer(ex.texts[0], return_tensors="pt", padding=False, truncation=True)
        labels = torch.zeros(tokenized["input_ids"].size(1), dtype=torch.long)

        for param, values in annotations.items():
            if param in param_to_idx:
                param_idx = param_to_idx[param]

                # 🔑 normalize values into a list
                if isinstance(values, str):
                    values = [values]
                elif not isinstance(values, (list, tuple)):
                    values = [str(values)]

                for value in values:
                    # safely encode each value
                    subword_ids = tokenizer.encode(str(value), add_special_tokens=False)
                    for tid in subword_ids:
                        mask = tokenized["input_ids"][0] == tid
                        labels[mask] = param_idx

        param_labels.append(labels)

    # pad labels so all sequences match
    max_len = max(len(labels) for labels in param_labels)
    param_labels = [F.pad(labels, (0, max_len - len(labels)), value=0) for labels in param_labels]
    param_labels = torch.stack(param_labels)

    return {"prompts": prompts, "contexts": contexts, "param_labels": param_labels}


# ===============================
# Extract Parameters from Prompt
# ===============================
def extract_parameters(model, tokenizer, prompt, param_to_idx, device="cpu"):
    model.eval()
    model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(inputs, return_sentence_embedding=True)
    param_logits = outputs["param_logits"][0]
    pred_indices = torch.argmax(F.softmax(param_logits, dim=-1), dim=-1).cpu().tolist()
    idx_to_param = {v: k for k, v in param_to_idx.items()}
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

    extracted = {}
    current_param = None
    current_tokens = []

    for tok, idx in zip(tokens, pred_indices):
        if idx == 0:
            if current_param and current_tokens:
                value = tokenizer.convert_tokens_to_string(current_tokens).strip()
                extracted.setdefault(current_param, []).append(value)
                current_param = None
                current_tokens = []
            continue
        param_name = idx_to_param[idx]
        if current_param != param_name:
            if current_param and current_tokens:
                value = tokenizer.convert_tokens_to_string(current_tokens).strip()
                extracted.setdefault(current_param, []).append(value)
            current_param = param_name
            current_tokens = [tok]
        else:
            current_tokens.append(tok)
    if current_param and current_tokens:
        value = tokenizer.convert_tokens_to_string(current_tokens).strip()
        extracted.setdefault(current_param, []).append(value)
    return extracted

# ===============================
# Load Base Model & Tokenizer
# ===============================
base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

data = load_data_from_jsonl("function_training_data.jsonl")
train_examples, param_to_idx = prepare_training_examples(data)
num_params = len(param_to_idx)

model = ParamExtractionModel(base_model, num_params=num_params)

train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=2,
    collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, param_to_idx)
)

# ===============================
# Training Loop
# ===============================
def train_model(model, dataloader, epochs=10, temperature=0.07, output_path="param_model.pt"):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    device = next(model.parameters()).device

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:  # <-- batching is already happening here
            optimizer.zero_grad()

            prompts = batch["prompts"]
            contexts = batch["contexts"]
            param_labels = batch["param_labels"].to(device)

            # Tokenize batch
            prompt_inputs = model.base_model.tokenize(prompts)
            context_inputs = model.base_model.tokenize(contexts)
            prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
            context_inputs = {k: v.to(device) for k, v in context_inputs.items()}

            # Forward pass
            prompt_outputs = model(prompt_inputs, return_sentence_embedding=True)
            context_outputs = model(context_inputs, return_sentence_embedding=True)

            prompt_embs = prompt_outputs["sentence_embedding"]  # shape [B, D]
            context_embs = context_outputs["sentence_embedding"]  # shape [B, D]
            param_logits = prompt_outputs["param_logits"]  # shape [B, num_params]

            # Losses
            contrast_loss = contrastive_loss(prompt_embs, context_embs, temperature)
            param_loss = F.cross_entropy(
                param_logits.view(-1, param_logits.size(-1)),
                param_labels.view(-1)
            )
            total = contrast_loss + param_loss

            # Backprop
            total.backward()
            optimizer.step()
            total_loss += total.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), output_path)
    print(f"✅ Training completed. Model saved to '{output_path}'.")

# ===============================
# Train the Model
# ===============================
# train_model(model, train_dataloader, epochs=10)

# ===============================
# Test Function Prediction + Param Extraction
# ===============================
test_prompt = "open firefox and navigate to kamingo.in"
extracted_params = extract_parameters(model, tokenizer, test_prompt, param_to_idx)
print("\nTest Prompt:", test_prompt)
print("Extracted Parameters:", extracted_params)
