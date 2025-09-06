import spacy
import spacy_transformers
from spacy.training import Example
from spacy.training import offsets_to_biluo_tags
import random
import argparse
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import uuid

# Parse command-line argument to choose model
parser = argparse.ArgumentParser(description="Fine-tune a model to detect parameters and chain functions with FAISS-based RAG")
parser.add_argument('--model', type=str, default='spacy',
                    help='Choose model to fine-tune: spacy or mpnet')
args = parser.parse_args()

# Simulated functions for chaining
def add_numbers(a: float, b: float) -> float:
    return a + b

def open_notepad_with_text(text: str) -> str:
    return f"Opened notepad with text: {text}"

# Load a blank English model with a transformer
nlp = spacy.blank("en")
nlp.add_pipe("transformer", config={"model": {"@architectures": "spacy-transformers.TransformerModel.v3", "name": "roberta-base"}})
ner = nlp.add_pipe("ner")
ner.add_label("PARAMETER")  # For numeric parameters
ner.add_label("URL_PARAMETER")  # For URL parameters
ner.add_label("FUNCTION")  # For function names

# Load training data from JSONL with error handling
TRAIN_DATA = []
with open('training_data.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        try:
            data = json.loads(line.strip())
            prompt = data.get("prompt", "")
            entities = data.get("entities", [])
            if prompt and entities:  # Only include valid entries
                TRAIN_DATA.append((prompt, entities))
            else:
                print(f"Skipping line {i}: Missing prompt or entities - {line.strip()}")
        except json.JSONDecodeError:
            print(f"Skipping line {i}: Invalid JSON - {line.strip()}")
        except Exception as e:
            print(f"Skipping line {i}: Error - {e} - {line.strip()}")

# Validate entity spans and convert to BILUO tags
def validate_annotations(nlp, train_data):
    for text, entities in train_data:
        doc = nlp.make_doc(text)
        try:
            tags = offsets_to_biluo_tags(doc, entities)
            if "-" in tags:
                tokens = [(token.idx, token.text) for token in doc]
                print(f"Tokens for '{text}': {tokens}")
                for start, end, label in entities:
                    for token_idx, token_text in tokens:
                        if abs(token_idx - start) <= 2 and abs((token_idx + len(token_text)) - end) <= 2:
                            print(f"Suggested span for '{text}' ({label}): ({token_idx}, {token_idx + len(token_text)}, '{label}')")
                raise ValueError(f"Invalid alignment in text '{text}': BILUO tags {tags}")
            print(f"Validated: {text} -> BILUO tags: {tags}")
        except Exception as e:
            print(f"Error validating '{text}': {e}")
            raise
    print("All annotations validated successfully.")

validate_annotations(nlp, TRAIN_DATA)

# Warm-up NER to initialize transitions
def warmup_ner(nlp, train_data):
    optimizer = nlp.begin_training()
    warmup_examples = [train_data[i] for i in [0, 5, 13, 14, 19, 22] if i < len(train_data)]  # Ensure indices are valid
    for text, entities in warmup_examples:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, {"entities": entities})
        nlp.update([example], drop=0.0, losses={}, sgd=optimizer)
    print("NER warmed up.")

warmup_ner(nlp, TRAIN_DATA)

# Train the NER model
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in ["ner", "transformer"]]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    optimizer.learn_rate = 0.00005
    for epoch in range(150):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, entities in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, {"entities": entities})
            try:
                nlp.update([example], drop=0.2, losses=losses, sgd=optimizer)
            except Exception as e:
                print(f"Error updating with text '{text}': {e}")
                print(f"Tokens: {[(token.idx, token.text) for token in doc]}")
                raise
        print(f"Epoch {epoch + 1} Losses: {losses}")

# Save fine-tuned model
nlp.to_disk("./fine_tuned_spacy_model")

# Initialize FAISS for RAG
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
vector_size = embedding_model.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(vector_size)  # L2 distance for similarity search
metadata_store = []  # Store metadata alongside embeddings

# Function to store result in FAISS
def store_result_in_faiss(prompt, function_name, params, result):
    context = f"Prompt: {prompt}, Function: {function_name}, Parameters: {params}, Result: {result}"
    embedding = embedding_model.encode(context).reshape(1, -1).astype('float32')
    point_id = len(metadata_store)
    faiss_index.add(embedding)
    metadata_store.append({
        "id": point_id,
        "prompt": prompt,
        "function": function_name,
        "params": params,
        "result": str(result),
        "context": context
    })
    print(f"Stored in FAISS: {context}")

# Function to retrieve parameter from FAISS
def retrieve_parameter_from_faiss(query, top_k=1):
    embedding = embedding_model.encode(query).reshape(1, -1).astype('float32')
    distances, indices = faiss_index.search(embedding, top_k)
    if indices[0][0] != -1 and distances[0][0] < 0.5:  # Threshold for relevance
        return metadata_store[indices[0][0]].get("result", None)
    return None

# Function to decompose prompt into sub-tasks
def decompose_prompt(doc):
    functions = []
    current_function = {"name": None, "params": [], "start_idx": None}
    for ent in doc.ents:
        if ent.label_ == "FUNCTION":
            if current_function["name"] is not None:
                functions.append(current_function)
            current_function = {"name": ent.text.lower(), "params": [], "start_idx": ent.start}
        elif ent.label_ in ["PARAMETER", "URL_PARAMETER"]:
            current_function["params"].append((ent.text, ent.label_))
    if current_function["name"] is not None:
        functions.append(current_function)
    return sorted(functions, key=lambda x: x["start_idx"] or 0)

# Function to map function names to callable functions
function_map = {
    "sum": add_numbers,
    "add": add_numbers,
    "write": open_notepad_with_text,
    "open": open_notepad_with_text,
}

# Function to execute chained functions with FAISS-based RAG
def execute_function_chain(prompt, doc, rag_enabled=True):
    functions = decompose_prompt(doc)
    prev_result = None
    results = []

    for func in functions:
        func_name = func["name"]
        params = [p[0] for p in func["params"] if p[1] == "PARAMETER"]
        url_params = [p[0] for p in func["params"] if p[1] == "URL_PARAMETER"]

        if func_name in ["sum", "add"]:
            if len(params) < 2 and rag_enabled:
                query = f"Result of {func_name} in prompt: {prompt}"
                rag_result = retrieve_parameter_from_faiss(query)
                if rag_result and len(params) == 1:
                    params.append(rag_result)
                    print(f"Retrieved missing parameter from FAISS: {rag_result}")

            if len(params) == 2:
                try:
                    params = [float(p) for p in params]
                    result = function_map[func_name](*params)
                    results.append({"function": func_name, "params": params, "result": result})
                    store_result_in_faiss(prompt, func_name, params, result)
                    prev_result = result
                except ValueError as e:
                    print(f"Error executing {func_name}: {e}")
            else:
                print(f"Insufficient parameters for {func_name}: {params}")

        elif func_name in ["write", "open"]:
            text_param = str(prev_result) if prev_result is not None else None
            if not text_param and rag_enabled:
                query = f"Result for {func_name} in prompt: {prompt}"
                text_param = retrieve_parameter_from_faiss(query)
                if text_param:
                    print(f"Retrieved text parameter from FAISS: {text_param}")

            if text_param:
                result = function_map[func_name](text_param)
                results.append({"function": func_name, "params": [text_param], "result": result})
                store_result_in_faiss(prompt, func_name, [text_param], result)
                prev_result = result
            else:
                print(f"No text parameter for {func_name}")

        elif url_params:
            result = f"URL action: {func_name} {url_params[0]}"
            results.append({"function": func_name, "params": url_params, "result": result})
            store_result_in_faiss(prompt, func_name, url_params, result)
            prev_result = result

    return results

# Test the model with chaining and FAISS-based RAG
nlp = spacy.load("./fine_tuned_spacy_model")
test_prompt = "open grok , youtube and google each in tabs in browser"
doc = nlp(test_prompt)
print(f"\nTesting prompt: {test_prompt}")
for ent in doc.ents:
    print(f"Parameter found: {ent.text} (label: {ent.label_})")

# Execute function chain
chain_results = execute_function_chain(test_prompt, doc)
print("\nPredicted function chain:")
for step in chain_results:
    params_str = ", ".join([str(p) for p in step["params"]])
    print(f"{step['function']}({params_str}) -> {step['result']}")

# Test with missing parameter (FAISS retrieval)
test_prompt_missing = "Write the sum in notepad."
doc_missing = nlp(test_prompt_missing)
print(f"\nTesting prompt with missing parameter: {test_prompt_missing}")
for ent in doc_missing.ents:
    print(f"Parameter found: {ent.text} (label: {ent.label_})")
chain_results_missing = execute_function_chain(test_prompt_missing, doc_missing)
print("\nPredicted function chain:")
for step in chain_results_missing:
    params_str = ", ".join([str(p) for p in step["params"]])
    print(f"{step['function']}({params_str}) -> {step['result']}")