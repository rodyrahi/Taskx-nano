import spacy
from sentence_transformers import SentenceTransformer, util
from functions import function_registry

# Load the fine-tuned spaCy model
model_path = "fine_tuned_spacy_model"
nlp = spacy.load(model_path)

# Load the all-mpnet-base-v2 model
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')



# Input dictionary
input_data = {
    "prompt": "search for cat videos and take a screenshot of the result"
}

def extract_parameters(prompt, function_name=None):
    """
    Extract parameters from a prompt using spaCy NER and all-mpnet-base-v2 similarity.
    Returns list of (clean_text, matched_type) tuples.
    """
    doc = nlp(prompt)
    parameters_with_types = []
    valid_labels = ["PARAMETER", "URL_PARAMETER", "TEXT_PARAMETER", "PLATFORM_PARAMETER", "FILE_PATH_PARAMETER", "PROGRAM_NAME_PARAMETER"]

    # Get expected parameter types (same as before)
    if function_name and function_name in function_registry:
        expected_param_types = function_registry[function_name].get("param_types", [])
    else:
        expected_param_types = ["URL", "NUMBER", "TEXT", "FILE_PATH", "PROGRAM"]

    # Examples for parameter types (unchanged)
    param_type_examples = {
        "URL": ["example.com", "kamingo.in", "https://google.com"],
        "NUMBER": [42, 100, 3.14, 5, 3],
        "TEXT": ["hello world", "search query", "browser", "notepad", "result", "cat videos"],
        "FILE_PATH": ["C:/path/to/file.txt", "D:/documents/report.docx", "/home/user/file.txt", "D:/startup/new_model/test.bat"],
        "PROGRAM": ["notepad", "chrome", "firefox", "stream", "outlook"],
    }

    # Flatten examples for encoding (unchanged)
    flat_examples = []
    example_labels = []
    for param_type, examples in param_type_examples.items():
        if param_type in expected_param_types or not expected_param_types:
            flat_examples.extend(examples)
            example_labels.extend([param_type] * len(examples))

    # Encode example parameters (unchanged)
    type_embeddings = model.encode(flat_examples, convert_to_tensor=True)

    # Iterate over ENTITIES (unchanged core logic)
    for ent in doc.ents:
        if ent.label_ not in valid_labels:
            continue

        clean_text = ent.text.strip(".,!?")
        
        if clean_text.lower() == "result":
            parameters_with_types.append(("__PREVIOUS_OUTPUT__", None))  # Special case, no type
            continue

        word_embedding = model.encode(clean_text, convert_to_tensor=True)
        similarities = util.cos_sim(word_embedding, type_embeddings)[0]

        max_similarity_idx = similarities.argmax()
        matched_type = example_labels[max_similarity_idx]
        max_similarity = similarities[max_similarity_idx]

        if ent.label_ == "FILE_PATH_PARAMETER":
            clean_text = clean_text.strip('"\'')
            clean_text = clean_text.replace(' ', '')

        print(f"Entity: {clean_text}, Label: {ent.label_}, "
              f"Matched Type: {matched_type}, Similarity: {max_similarity}")

        if max_similarity > 0.4 and (not expected_param_types or matched_type in expected_param_types):
            parameters_with_types.append((clean_text, matched_type))

    return parameters_with_types

def process_input(input_data):
    """
    Process the input prompt to identify distinct commands and select chained functions.
    """
    prompt = input_data["prompt"]
    doc = nlp(prompt)
    
    # Rule-based command segmentation (unchanged)
    commands = []
    current_command = []
    conjunctions = {"and", "then"}
    
    for token in doc:
        current_command.append(token.text)
        if token.text.lower() in conjunctions and current_command:
            next_tokens = [t.text for t in doc[token.i + 1:]]
            next_phrase = " ".join(next_tokens)
            next_doc = nlp(next_phrase)
            has_function = any(t.ent_type_ == "FUNCTION" for t in next_doc)
            if has_function:
                commands.append(" ".join(current_command[:-1]).strip())
                current_command = [token.text]
    
    if current_command:
        commands.append(" ".join(current_command).strip())
    
    commands = [cmd for cmd in commands if cmd]
    print(f"Identified commands: {commands}")
    
    # Precompute description embeddings
    function_names = list(function_registry.keys())
    description_embeddings = model.encode(
        [info["description"] for info in function_registry.values()],
        convert_to_tensor=True
    )
    
    result_chain = []
    for i, command in enumerate(commands):
        # Extract parameters and types first
        params_with_types = extract_parameters(command, function_name=None)
        detected_params = [p for p, t in params_with_types]
        detected_types = [t for p, t in params_with_types if t is not None]
        print(f"Detected params: {detected_params}, Detected types: {detected_types}")
        
        # Get function candidates from spaCy
        command_doc = nlp(command)
        function_candidates = [token.text for token in command_doc if token.ent_type_ == "FUNCTION"]
        function_query = " ".join(function_candidates) if function_candidates else command
        
        # Encode the command query
        command_embedding = model.encode(function_query, convert_to_tensor=True)
        
        # Compute similarities and type-matching scores
        similarities = util.cos_sim(command_embedding, description_embeddings)[0]
        type_match_weight = 0.8
        type_mismatch_penalty = -0.5
        
        best_score = -float('inf')
        best_function_name = None
        
        for idx, func_name in enumerate(function_names):
            expected_param_types = function_registry[func_name].get("param_types", [])
            print(f"Function: {func_name}, Expected types: {expected_param_types}")
            
            # Calculate type match fraction
            if detected_types and expected_param_types:
                matching_types = sum(1 for dt in detected_types if dt in expected_param_types)
                type_match_fraction = matching_types / len(detected_types)
            elif detected_types:
                type_match_fraction = 0.0
            elif expected_param_types:
                type_match_fraction = 0.2
            else:
                type_match_fraction = 0.5
            
            total_score = similarities[idx].item() + (type_match_fraction * type_match_weight)
            if detected_types and expected_param_types and not any(dt in expected_param_types for dt in detected_types):
                total_score += type_mismatch_penalty
            
            print(f"Function: {func_name}, Similarity: {similarities[idx]:.3f}, Type Match: {type_match_fraction:.3f}, Total Score: {total_score:.3f}")
            
            if total_score > best_score and total_score > 0.5:  # Lowered threshold
                best_score = total_score
                best_function_name = func_name
        
        if not best_function_name:
            print(f"No suitable function found for command: {command}")
            continue
        
        # Re-extract parameters with the selected function's expected types
        final_params_with_types = extract_parameters(command, function_name=best_function_name)
        step_params = [p for p, t in final_params_with_types]
        
        # Validate parameter count and types
        required_params = function_registry[best_function_name]["required_params"]
        expected_types = function_registry[best_function_name]["param_types"]
        if len(step_params) < len(required_params):
            print(f"Error: Insufficient parameters for {best_function_name}. Required: {required_params}, Provided: {step_params}")
            continue
        if expected_types and detected_types and not all(t in expected_types for t in detected_types):
            print(f"Error: Type mismatch for {best_function_name}. Expected: {expected_types}, Detected: {detected_types}")
            continue
        
        print(f"Command: {command}, Function: {best_function_name}, Parameters: {step_params}")
        
        result_chain.append({
            "function_to_use": best_function_name,
            "parameters": step_params
        })
    
    return result_chain

def execute_chain(result_chain):
    """
    Execute the chain of functions with their parameters.
    
    Args:
        result_chain (list): List of dictionaries with function names and parameters.
    
    Returns:
        The final output of the chain.
    """
    previous_output = None
    for step in result_chain:
        function_name = step["function_to_use"]
        params = step["parameters"]
        selected_function = function_registry[function_name]["function"]
        
        # Replace __PREVIOUS_OUTPUT__ with the actual previous output
        resolved_params = [previous_output if p == "__PREVIOUS_OUTPUT__" else p for p in params]
        
        # Execute the function
        output = selected_function(*resolved_params)
        previous_output = output
    
    return previous_output

# Process the input
result_chain = process_input(input_data)

# Execute the chain if no error
if isinstance(result_chain, dict) and "error" in result_chain:
    print(result_chain)
else:
    print(f"Input: {input_data}")
    print(f"Model output: {result_chain}")
    final_output = execute_chain(result_chain)
    print(f"Final result: {final_output}")