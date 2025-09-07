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
    "prompt": "open steam on my computer"
}

def extract_parameters(prompt, function_name=None):
    """
    Extract parameters from a prompt using spaCy NER and all-mpnet-base-v2 similarity.
    """
    doc = nlp(prompt)
    parameters = []
    valid_labels = ["PARAMETER", "URL_PARAMETER", "TEXT_PARAMETER", "PLATFORM_PARAMETER" , "FILE_PATH_PARAMETER", "PROGRAM_NAME_PARAMETER"]

    # Get expected parameter types
    if function_name and function_name in function_registry:
        expected_param_types = function_registry[function_name].get("param_types", [])
    else:
        expected_param_types = ["URL", "NUMBER", "TEXT" , "FILE_PATH", "PROGRAM"]

    # Examples for parameter types
    param_type_examples = {
        "URL": ["example.com", "kamingo.in", "https://google.com"],
        "NUMBER": [42, 100, 3.14, 5, 3],
        "TEXT": ["hello world", "search query", "browser", "notepad", "result", "cat videos" ] ,
        "FILE_PATH": ["C:/path/to/file.txt", "D:/documents/report.docx", "/home/user/file.txt", "D:/startup/new_model/test.bat"],
        "PROGRAM": ["notepad", "chrome", "firefox", "stream", "outlook"],

    }

    # Flatten examples for encoding
    flat_examples = []
    example_labels = []
    for param_type, examples in param_type_examples.items():
        if param_type in expected_param_types or not expected_param_types:
            flat_examples.extend(examples)
            example_labels.extend([param_type] * len(examples))

    # Encode example parameters
    type_embeddings = model.encode(flat_examples, convert_to_tensor=True)

    # 🔑 Iterate over ENTITIES (spans), not tokens
    for ent in doc.ents:
        if ent.label_ not in valid_labels:
            continue

        clean_text = ent.text.strip(".,!?")

        if clean_text.lower() == "result":
            parameters.append("__PREVIOUS_OUTPUT__")
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
            parameters.append(clean_text)

    return parameters

def process_input(input_data):
    """
    Process the input prompt to identify distinct commands and select chained functions.
    
    Args:
        input_data (dict): Input dictionary containing the prompt.
    
    Returns:
        list: A chain of dictionaries specifying functions and their parameters.
    """
    prompt = input_data["prompt"]
    doc = nlp(prompt)
    
    # Initialize result chain
    result_chain = []
    
    # Rule-based command segmentation
    commands = []
    current_command = []
    conjunctions = {"and", "then"}
    
    # Improved segmentation: Avoid splitting on conjunctions unless a clear function follows
    for token in doc:
        current_command.append(token.text)
        
        # Check if the current token sequence forms a valid command
        if token.text.lower() in conjunctions and current_command:
            # Look ahead to ensure the conjunction is followed by a potential function
            next_tokens = [t.text for t in doc[token.i + 1:]]
            next_phrase = " ".join(next_tokens)
            next_doc = nlp(next_phrase)
            has_function = any(t.ent_type_ == "FUNCTION" for t in next_doc)
            
            if has_function:
                commands.append(" ".join(current_command[:-1]).strip())
                current_command = [token.text]
    
    # Append the last command
    if current_command:
        commands.append(" ".join(current_command).strip())
    
    # Remove empty commands
    commands = [cmd for cmd in commands if cmd]
    print(f"Identified commands: {commands}")
    
    # Process each command
    for i, command in enumerate(commands):
        # Process command with spaCy to identify function names
        command_doc = nlp(command)
        function_candidates = [token.text for token in command_doc if token.ent_type_ == "FUNCTION"]
        
        # Use function candidates or entire command for matching
        function_query = " ".join(function_candidates) if function_candidates else command
        
        # Encode the command or function query
        command_embedding = model.encode(function_query, convert_to_tensor=True)
        description_embeddings = model.encode(
            [info["description"] for info in function_registry.values()],
            convert_to_tensor=True
        )
        
        # Compute cosine similarity
        similarities = util.cos_sim(command_embedding, description_embeddings)[0]
        
        # Find the function with the highest similarity
        max_similarity_idx = similarities.argmax()
        best_function_name = list(function_registry.keys())[max_similarity_idx]
        
        # Extract parameters for this specific command
        parameters = extract_parameters(command, function_name=best_function_name)
        print(f"Command: {command}, Function: {best_function_name}, Parameters: {parameters}")
        
        # Assign parameters based on position in the chain
        step_params = parameters
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