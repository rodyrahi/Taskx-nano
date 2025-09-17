import faiss
import spacy
from sentence_transformers import SentenceTransformer, util
from reranker import rerank_functions  # Assuming this is defined elsewhere
from functions import function_registry  # Assuming this is defined elsewhere

# Sample function_registry for demonstration
# function_registry = {
#     "call_llm": {
#         "description": "Generate text using a language model with a given prompt",
#         "param_types": ["PROMPT"],
#         "required_params": ["prompt"],
#         "function": lambda prompt: f"Generated poem: Roses are red, violets are blue, {prompt} is fun, and so are you!"
#     },
#     "open_notepad": {
#         "description": "Write text to notepad or a temporary text file",
#         "param_types": ["TEXT"],
#         "required_params": ["text"],
#         "function": lambda text: f"Text written to editor: C:\\Users\\Temp\\notepad_output.txt with content: {text}"
#     }
# }

# Load the fine-tuned spaCy model
model_path = "fine_tuned_spacy_model"
nlp = spacy.load(model_path)

# Load the multi-qa-mpnet-base-dot-v1 model
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Input dictionary
input_data = {
    "prompt": "write a poem and then put the result in notepad"
}

functions_score = []


previous_results = []


def extract_parameters(prompt, function_name=None):
    """
    Extract parameters from a prompt using spaCy NER and multi-qa-mpnet-base-dot-v1 similarity.
    Returns list of (clean_text, matched_type) tuples.
    """
    doc = nlp(prompt)
    parameters_with_types = []
    valid_labels = ["PARAMETER", "URL_PARAMETER", "TEXT_PARAMETER", "PLATFORM_PARAMETER", "FILE_PATH_PARAMETER", "PROGRAM_NAME_PARAMETER", "PROMOT_PARAMETER"]

    # Get expected parameter types
    if function_name and function_name in function_registry:
        expected_param_types = function_registry[function_name].get("param_types", [])
    else:
        expected_param_types = ["URL", "NUMBER", "TEXT", "FILE_PATH", "PROGRAM", "PROMPT"]

    # Combined and enhanced examples for parameter types
    param_type_examples = {
        "URL": ["example.com", "kamingo.in", "https://google.com"],
        "NUMBER": [42, 100, 3.14, 5 ,3],
        "TEXT": ["hey there how are you", "hello world"],
        "FILE_PATH": ["C:/path/to/file.txt", "D:/documents/report.docx", "/home/user/file.txt", "D:/startup/new_model/test.bat"],
        "PROGRAM": ["open notepad", "launch chrome", "start firefox", "run stream", "use outlook", "notepad", "chrome", "firefox", "stream", "outlook"],
        "PROMPT": ["write a poem", "draft an email", "create a story", "compose a message"]
    }

    # Flatten examples for encoding
    flat_examples = []
    example_labels = []
    for param_type, examples in param_type_examples.items():
        if param_type in expected_param_types or not expected_param_types:
            flat_examples.extend([str(ex) for ex in examples])
            example_labels.extend([param_type] * len(examples))

    # Encode example parameters using the model
    type_embeddings = model.encode(flat_examples, convert_to_tensor=True)

    # Process entities
    for ent in doc.ents:
        if ent.label_ not in valid_labels:
            continue

        clean_text = ent.text.strip(".,!?")

        # # Extract descriptions (or function names if descriptions are unavailable)
        # descriptions = [item for item in previous_results]

        # # Generate embeddings
        # embeddings = model.encode(descriptions, convert_to_numpy=True)

        # # Initialize FAISS index
        # dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
        # index = faiss.IndexFlatL2(dimension)  # L2 distance index
        # index.add(embeddings)  # Add embeddings to the index

        # # Store metadata (function, parameters, output, type)
        # metadata = previous_results

        
        
        if clean_text.lower() == "result":
            parameters_with_types.append(("__PREVIOUS_OUTPUT__", None))
            print(f"Detected 'result' entity, mapped to __PREVIOUS_OUTPUT__")
            continue

        # Use the model to get contextual embedding for better similarity
        word_embedding = model.encode(clean_text, convert_to_tensor=True)
        similarities = util.cos_sim(word_embedding, type_embeddings)[0]

        max_similarity_idx = similarities.argmax()
        matched_type = example_labels[max_similarity_idx]
        max_similarity = similarities[max_similarity_idx]

        if ent.label_ == "FILE_PATH_PARAMETER":
            clean_text = clean_text.strip('"\'')
            clean_text = clean_text.replace(' ', '')

        print(f"Entity: {clean_text}, Label: {ent.label_}, "
              f"Matched Type: {matched_type}, Similarity: {max_similarity:.3f}")

        if max_similarity > 0.4 and (not expected_param_types or matched_type in expected_param_types):
            parameters_with_types.append((clean_text, matched_type))

    return parameters_with_types

def find_best_function_bge(command, detected_types=None):
    """
    Use multi-qa-mpnet-base-dot-v1 to find the best matching function for a command.
    """
    function_names = list(function_registry.keys())
    description_embeddings = model.encode(
        [str(info["description"]) + f" with parameters type: {info['param_types']} and required parameters: {info['required_params']}" for info in function_registry.values()],
        convert_to_tensor=True
    )

    print("\nFunction descriptions and metadata:")
    for info in function_registry.values():
        print(str(info["description"]) + f" with parameters type: {info['param_types']} and required parameters: {info['required_params']}")
    print("-----")
    
    # Create enhanced query with parameter context
    if detected_types:
        type_context = f" with parameters of types: [{', '.join(detected_types)}]"
    else:
        type_context = ""
    
    enhanced_query = f"{command}{type_context}"
    print(f"Enhanced query for function matching: '{enhanced_query}'")
    
    query_embedding = model.encode(enhanced_query, convert_to_tensor=True)
    
    # Compute similarities
    similarities = util.cos_sim(query_embedding, description_embeddings)[0]
    
    # Type matching enhancement
    type_match_weight = 0.3
    type_mismatch_penalty = -0.3
    similarity_threshold = 0.6
    
    best_score = -float('inf')
    best_function_name = None
    
    for idx, func_name in enumerate(function_names):
        func_info = function_registry[func_name]
        expected_param_types = func_info.get("param_types", [])
        
        sim_score = similarities[idx].item()
        
        if detected_types and expected_param_types:
            matching_types = sum(1 for dt in detected_types if dt in expected_param_types)
            type_match_fraction = matching_types / len(detected_types)
            coverage_fraction = sum(1 for et in expected_param_types 
                                  if et in detected_types) / len(expected_param_types) if expected_param_types else 1.0
            type_score = (type_match_fraction + coverage_fraction) / 2
        elif detected_types:
            type_score = 0.1
        elif expected_param_types:
            type_score = 0.3
        else:
            type_score = 0.8
        
        total_score = sim_score + (type_score * type_match_weight)
        
        if detected_types and expected_param_types and not any(dt in expected_param_types for dt in detected_types):
            total_score += type_mismatch_penalty
        
        # print(f"Function: {func_name}, Sim: {sim_score:.3f}, Type: {type_score:.3f}, Total: {total_score:.3f}")
        functions_score.append((func_name, func_info["description"], total_score))

    # Sort functions_score by total_score in descending order
    sorted_functions_score = sorted(functions_score, key=lambda x: x[2], reverse=True)
    
    # Placeholder for rerank_functions (assuming it returns a list of dicts with 'name' and 'adjusted_score')
    reranks = rerank_functions(sorted_functions_score, command) if 'rerank_functions' in globals() else [{'name': sorted_functions_score[0][0], 'adjusted_score': sorted_functions_score[0][2]}]
    
    best_function_name = reranks[0]['name']
    best_score = reranks[0]['adjusted_score']


    print(f"\nSelected function from reranking: {best_function_name} with score: {best_score:.3f}\n")
    
    return best_function_name, best_score

def process_input(input_data):
    """
    Process the input prompt using multi-qa-mpnet-base-dot-v1 for function selection.
    """
    prompt = input_data["prompt"]
    doc = nlp(prompt)
    
    # Rule-based command segmentation
    commands = []
    current_command = []
    conjunctions = {"and", "then", "after"}
    
    for token in doc:
        current_command.append(token.text)
        if token.text.lower() in conjunctions and current_command:
            next_tokens = [t.text for t in doc[token.i + 1:]]
            if next_tokens:
                next_phrase = " ".join(next_tokens)
                next_doc = nlp(next_phrase)
                has_function = any(t.ent_type_ == "FUNCTION" for t in next_doc)
                if has_function:
                    commands.append(" ".join(current_command[:-1]).strip())
                    current_command = []
    
    if current_command:
        commands.append(" ".join(current_command).strip())
    
    commands = [cmd for cmd in commands if cmd]
    print(f"Identified commands: {commands}")
    
    result_chain = []
    
    for i, command in enumerate(commands):
        params_with_types = extract_parameters(command, function_name=None)
        detected_params = [p for p, t in params_with_types]
        detected_types = [t for p, t in params_with_types if t is not None]
        print(f"Command '{command}': Detected params: {detected_params}, Types: {detected_types}")
        
        best_function_name, best_score = find_best_function_bge(command, detected_types)
        
        if not best_function_name:
            print(f"No suitable function found for command: '{command}' (best score: {best_score:.3f})")
            continue
        
        # print(f"Selected function: {best_function_name} (score: {best_score:.3f})")
        
        # Re-extract parameters with the selected function's expected types
        final_params_with_types = extract_parameters(command, function_name=best_function_name)
        step_params = [p for p, t in final_params_with_types]
        
        func_info = function_registry[best_function_name]
        required_params = func_info.get("required_params", [])
        expected_types = func_info.get("param_types", [])
        
        if not required_params and not expected_types:
            step_params = []
        
        if len(step_params) < len(required_params):
            print(f"Warning: Insufficient parameters for {best_function_name}. "
                  f"Required: {len(required_params)}, Provided: {len(step_params)}")
            continue
        
        if expected_types and detected_types:
            valid_types = all(t in expected_types for t in detected_types if t is not None)
            if not valid_types:
                print(f"Warning: Type mismatch for {best_function_name}. "
                      f"Expected: {expected_types}, Detected: {detected_types}")
                continue
        
        print(f"Final mapping - Command: '{command}' -> Function: {best_function_name}, "
              f"Parameters: {step_params}")
        
        result_chain.append({
            "function_to_use": best_function_name,
            "parameters": step_params
        })
    
    return result_chain

def execute_chain(result_chain):
    """
    Execute the chain of functions with their parameters.
    """
    if not result_chain:
        return {"error": "No valid function chain found."}
    
    previous_output = None
    execution_results = []
    
    for i, step in enumerate(result_chain):
        try:
            function_name = step["function_to_use"]
            params = step["parameters"]
            selected_function = function_registry[function_name]["function"]
            
            # Replace __PREVIOUS_OUTPUT__ with actual previous output
            resolved_params = [previous_output if p == "__PREVIOUS_OUTPUT__" else p for p in params]
            
            print(f"Executing step {i+1}: {function_name}({', '.join(map(str, resolved_params))})")
            
            output = selected_function(*resolved_params)

            p = {
                "function": function_name,
                "parameters": resolved_params,
                "output": output ,
                "type": "text"
            }
            previous_results.append(p)

            previous_output = output
            execution_results.append(f"Step {i+1}: {output}")
            
        except Exception as e:
            error_msg = f"Error executing {function_name}: {str(e)}"
            print(error_msg)
            execution_results.append(error_msg)
            break
    
    return "\n".join(execution_results) if execution_results else "Execution completed with no output."

if __name__ == "__main__":
    print("Processing input with multi-qa-mpnet-base-dot-v1 function matching...")
    print(f"Input: {input_data}")
    
    result_chain = process_input(input_data)
    
    if isinstance(result_chain, dict) and "error" in result_chain:
        print(result_chain["error"])
    elif not result_chain:
        print("No valid function chain generated.")
    else:
        print(f"Generated chain: {result_chain}")
        final_output = execute_chain(result_chain)
        print(f"\nFinal result:\n{final_output}")