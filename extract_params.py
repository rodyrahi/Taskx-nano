from sentence_transformers import SentenceTransformer, util
import spacy
from functions import function_registry

# Load spaCy model
nlp = spacy.load("fine_tuned_spacy_model")

# Load SentenceTransformer model
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")


def extract_parameters(prompt: str, debug: bool = False):
    """
    Extract parameters from a prompt using spaCy NER + semantic similarity.

    Args:
        prompt (str): Input text prompt.
        debug (bool): Print debug logs.

    Returns:
        List[Tuple[str, str]]: List of (clean_text, matched_type) tuples.
    """
    doc = nlp(prompt)
    parameters_with_types = []

    valid_labels = {
        "PARAMETER", "URL_PARAMETER", "TEXT_PARAMETER",
        "PLATFORM_PARAMETER", "FILE_PATH_PARAMETER",
        "PROGRAM_NAME_PARAMETER", "PROMOT_PARAMETER"
    }

    # Collect ALL expected param types from the registry
    expected_param_types = set()
    for f_data in function_registry.values():
        expected_param_types.update(f_data.get("param_types", []))
    if not expected_param_types:
        expected_param_types = {"URL", "NUMBER", "TEXT", "FILE_PATH", "PROGRAM", "PROMPT"}

    # Examples for parameter types
    param_type_examples = {
        "URL": ["example.com", "https://google.com"],
        "NUMBER": ["42", "100", "3.14"],
        "TEXT": ["hey there how are you", "hello world"],
        "FILE_PATH": [
            "C:/path/to/file.txt", "D:/documents/report.docx",
            "/home/user/file.txt", "D:/startup/new_model/test.bat"
        ],
        "PROGRAM": ["notepad", "chrome", "firefox", "outlook", "stream"],
        "PROMPT": ["write a poem", "draft an email", "create a story", "compose a message"]
    }

    # Flatten examples for encoding
    flat_examples, example_labels = [], []
    for param_type, examples in param_type_examples.items():
        if param_type in expected_param_types:
            flat_examples.extend(examples)
            example_labels.extend([param_type] * len(examples))

    # Encode reference examples once
    type_embeddings = model.encode(flat_examples, convert_to_tensor=True)

    # Process entities
    for ent in doc.ents:
        if ent.label_ not in valid_labels:
            continue

        clean_text = ent.text.strip(".,!?").strip()

        # Special case → use previous output
        if clean_text.lower() == "result":
            parameters_with_types.append(("__PREVIOUS_OUTPUT__", None))
            if debug:
                print("Detected 'result' → mapped to __PREVIOUS_OUTPUT__")
            continue

        # Compare entity embedding to reference examples
        word_embedding = model.encode(clean_text, convert_to_tensor=True)
        similarities = util.cos_sim(word_embedding, type_embeddings)[0]

        max_idx = similarities.argmax().item()
        matched_type = example_labels[max_idx]
        max_score = similarities[max_idx].item()

        # Special cleaning for file paths
        if ent.label_ == "FILE_PATH_PARAMETER":
            clean_text = clean_text.strip('"\'').replace(" ", "")

        if debug:
            print(f"Entity: {clean_text}, Label: {ent.label_}, "
                  f"Matched Type: {matched_type}, Score: {max_score:.3f}")

        # Keep only good matches
        if max_score > 0.4:
            parameters_with_types.append((clean_text, matched_type))

    return parameters_with_types
