import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# -------- Dataset --------
class PromptFunctionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        func_text = f"{item['function_signature']} | {item['context']} | {item['args']}"
        return prompt, func_text

# -------- Text Encoder --------
class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", embed_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, embed_dim)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:,0,:]  # CLS token
        embedding = self.projection(pooled)
        return F.normalize(embedding, dim=-1)

# -------- Contrastive Loss --------
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size = z1.size(0)
        sim = z1 @ z2.T / self.temperature
        labels = torch.arange(batch_size).to(z1.device)
        loss_i = nn.CrossEntropyLoss()(sim, labels)
        loss_j = nn.CrossEntropyLoss()(sim.T, labels)
        return (loss_i + loss_j) / 2

# -------- Sample Data --------
data = [
    {"prompt": "play the song 'Shape of You' by Ed Sheeran",
     "function_signature": "play_music(song: str, artist: str)",
     "context": "Plays a song by the specified artist in the music player.",
     "args": {"song": "Shape of You", "artist": "Ed Sheeran"}},
    {"prompt": "set an alarm for 7:00 AM",
     "function_signature": "set_alarm(time: str)",
     "context": "Sets an alarm for the given time.",
     "args": {"time": "7:00 AM"}}
]

# -------- Tokenizer & Dataset --------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = PromptFunctionDataset(data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# -------- Models --------
prompt_encoder = TextEncoder()
func_encoder = TextEncoder()
optimizer = torch.optim.Adam(list(prompt_encoder.parameters()) + list(func_encoder.parameters()), lr=1e-5)
loss_fn = ContrastiveLoss()

# -------- Training --------
prompt_encoder.train()
func_encoder.train()
for epoch in range(10):
    for prompts, funcs in dataloader:
        prompt_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        func_inputs = tokenizer(funcs, padding=True, truncation=True, return_tensors="pt")

        z1 = prompt_encoder(prompt_inputs["input_ids"], prompt_inputs["attention_mask"])
        z2 = func_encoder(func_inputs["input_ids"], func_inputs["attention_mask"])

        loss = loss_fn(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

# -------- Testing --------
prompt_encoder.eval()
func_encoder.eval()

# Test prompt
test_prompt = "open notepad and write Hello World in it"
test_input = tokenizer([test_prompt], padding=True, truncation=True, return_tensors="pt")
z_test = prompt_encoder(test_input["input_ids"], test_input["attention_mask"])

# Encode all function options
func_texts = [f"{d['function_signature']} | {d['context']} | {d['args']}" for d in data]
func_inputs = tokenizer(func_texts, padding=True, truncation=True, return_tensors="pt")
z_funcs = func_encoder(func_inputs["input_ids"], func_inputs["attention_mask"])

# Cosine similarity
cos_sim = z_test @ z_funcs.T
best_idx = torch.argmax(cos_sim, dim=-1).item()
pred_func = data[best_idx]['function_signature']
pred_args = data[best_idx]['args']

print("\nTest Prompt:", test_prompt)
print("Predicted Function:", pred_func)
print("Predicted Args:", pred_args)
