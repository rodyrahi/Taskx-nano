# vector_functions_runtime.py
import os
import re
import json
import hashlib
import inspect
from typing import Dict, Any, List, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# ---- your stuff ----
from testing_funcs.funcs import FUNCTION_REGISTRY, call_function
from train import ParamExtractionModel, extract_parameters  # must exist

# ===============================
# Config
# ===============================
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "function_registry.index"
META_PATH = "function_registry.json"
PARAM_MODEL_PATH = "param_model.pt"
PARAM_TO_IDX_PATH = "param_to_idx.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# Helpers
# ===============================
def _signature_info(fn) -> Tuple[List[str], List[str]]:
    """Return ordered positional args and keyword-only args."""
    sig = inspect.signature(fn)
    pos_or_kw = []
    kw_only = []
    for name, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD):
            pos_or_kw.append(name)
        elif p.kind == inspect.Parameter.KEYWORD_ONLY:
            kw_only.append(name)
    return pos_or_kw, kw_only

def _registry_snapshot() -> List[Dict[str, Any]]:
    """Freeze FUNCTION_REGISTRY into a serializable list with arg order."""
    items = []
    for name, data in FUNCTION_REGISTRY.items():
        fn = data["function"]
        pos_args, kwonly_args = _signature_info(fn)
        items.append({
            "name": name,
            "signature": data.get("signature", ""),
            "doc": data.get("doc", "") or "",
            "module": getattr(fn, "__module__", ""),
            "args": pos_args,
            "kwonly_args": kwonly_args,
        })
    # keep deterministic order
    items.sort(key=lambda x: x["name"])
    return items

def _entry_text(entry: Dict[str, Any]) -> str:
    """Text used for embedding a function."""
    return f"{entry['name']} | {entry['signature']} | {entry['doc']} | args={entry['args']} | module={entry['module']}"

def _digest(entries: List[Dict[str, Any]]) -> str:
    blob = "\n".join(_entry_text(e) for e in entries)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

# ===============================
# Build / Load Index
# ===============================
def build_or_load_index(force_rebuild: bool = False):
    base_model = SentenceTransformer(EMBED_MODEL)

    entries = _registry_snapshot()
    current_digest = _digest(entries)

    if (not force_rebuild) and os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("digest") == current_digest and meta.get("embed_model") == EMBED_MODEL:
            index = faiss.read_index(INDEX_PATH)
            return base_model, index, meta["entries"]

    # Build fresh
    texts = [_entry_text(e) for e in entries]
    embs = base_model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embs)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "embed_model": EMBED_MODEL,
            "digest": current_digest,
            "entries": entries
        }, f, indent=2)

    return base_model, index, entries

# ===============================
# Load Parameter Model
# ===============================
def load_param_model():
    base_model_for_params = SentenceTransformer(EMBED_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    with open(PARAM_TO_IDX_PATH, "r", encoding="utf-8") as f:
        param_to_idx = json.load(f)
    num_params = len(param_to_idx)
    model = ParamExtractionModel(base_model_for_params, num_params)
    model.load_state_dict(torch.load(PARAM_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, tokenizer, param_to_idx

# ===============================
# Predict best function
# ===============================
def predict_best_function(prompt: str, base_model: SentenceTransformer, index, entries, top_k: int = 1):
    emb = base_model.encode([prompt], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    D, I = index.search(emb, k=top_k)
    best_idx = int(I[0][0])
    best_entry = entries[best_idx]
    score = float(D[0][0])
    return best_entry, score

# ===============================
# Extract + order params
# ===============================
def extract_and_align_params(prompt: str, best_entry: Dict[str, Any],
                             param_model, tokenizer, param_to_idx) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Return (ordered_args, kwonly_kwargs) aligned to function signature.
    """
    extracted = extract_parameters(param_model, tokenizer, prompt, param_to_idx, device=DEVICE)
    extracted = extracted or {}

    # Positional args in order
    ordered_args = []
    for name in best_entry["args"]:
        if name in extracted:
            ordered_args.append(extracted[name])
        # If missing and function likely has a default, it's fine to skip (Python will use default)

    # Keyword-only kwargs
    kwonly_kwargs = {k: extracted[k] for k in best_entry["kwonly_args"] if k in extracted}
    return ordered_args, kwonly_kwargs

# ===============================
# Predict + run
# ===============================
def predict_and_run(prompt: str, top_k: int = 1):
    # 1) Ensure FAISS index reflects current FUNCTION_REGISTRY
    base_model, index, entries = build_or_load_index()

    # 2) Find best function
    best_entry, score = predict_best_function(prompt, base_model, index, entries, top_k=top_k)

    # 3) Load param model & extract params
    param_model, tokenizer, param_to_idx = load_param_model()
    ordered_args, kwonly_kwargs = extract_and_align_params(prompt, best_entry, param_model, tokenizer, param_to_idx)

    # 4) Call real function
    name = best_entry["name"]
    try:
        result = call_function(name, *ordered_args[0], **kwonly_kwargs)
        return {
            "prompt": prompt,
            "best_match": best_entry,
            "similarity": score,
            "ordered_args": ordered_args,
            "kwonly_kwargs": kwonly_kwargs,
            "result": result
        }
    except Exception as e:
        return {
            "prompt": prompt,
            "best_match": best_entry,
            "similarity": score,
            "ordered_args": ordered_args,
            "kwonly_kwargs": kwonly_kwargs,
            "error": str(e)
        }

# ===============================
# CLI / Demo
# ===============================
if __name__ == "__main__":
    # Example prompt
    prompt = "write 'hey there how are you doing' in the default text editor"

    out = predict_and_run(prompt)
    print("\n--- Prediction & Execution ---")
    print("Prompt:", out["prompt"])
    print("Best Function:", out["best_match"]["signature"])
    print("Similarity:", out["similarity"])
    print("Ordered Args:", out["ordered_args"])
    if out["kwonly_kwargs"]:
        print("KW-only Kwargs:", out["kwonly_kwargs"])
    if "result" in out:
        print("Result:", out["result"])
    else:
        print("Execution Error:", out["error"])
