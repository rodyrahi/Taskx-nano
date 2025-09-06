import spacy
from sentence_transformers import SentenceTransformer, util

# Load the fine-tuned spaCy model
model_path = "fine_tuned_spacy_model"
nlp = spacy.load(model_path)

# Load the all-mpnet-base-v2 model
model = SentenceTransformer('all-mpnet-base-v2')

# Define possible functions and their descriptions
function_descriptions = {
    "add": "Add two numbers together",
    "subtract": "Subtract one number from another",
    "multiply": "Multiply two numbers",
    "divide": "Divide one number by another",
    "write_in_notepad": "Write text or a number to notepad"
}

# Function implementations
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b != 0:
        return a / b
    else:
        return "Error: Division by zero"

def write_in_notepad(text):
    return f"{text} added to notepad"

# Map function names to their implementations
function_map = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
    "write_in_notepad": write_in_notepad
}

# Input dictionary
input_data = {
    "prompt": "Sum 5 plus 10 and write it in notepad"
}

# Function to extract parameters using spaCy
def extract_parameters(prompt):
    doc = nlp(prompt)
    parameters = [int(ent.text) for ent in doc.ents if ent.label_ == "PARAMETER"]
    return parameters

# Function to process input and select chained functions
def process_input(input_data):
    prompt = input_data["prompt"]
    
    # Split the prompt at "and" to identify potential chained functions
    prompt_parts = [part.strip() for part in prompt.split(" and ")]
    result_chain = []

    # Extract parameters from the entire prompt
    parameters = extract_parameters(prompt)
    
    # Validate parameters for the first function
    if len(parameters) != 2:
        return {"error": "Please provide exactly two parameters for the mathematical operation."}

    # Process each part of the prompt
    for i, part in enumerate(prompt_parts):
        # Encode the prompt part
        prompt_embedding = model.encode(part, convert_to_tensor=True)
        description_embeddings = model.encode(list(function_descriptions.values()), convert_to_tensor=True)

        # Compute cosine similarity
        similarities = util.cos_sim(prompt_embedding, description_embeddings)[0]

        # Find the function with the highest similarity
        max_similarity_idx = similarities.argmax()
        best_function_name = list(function_descriptions.keys())[max_similarity_idx]

        # Assign parameters based on the function and position in the chain
        if i == 0:
            # First function uses extracted parameters
            result_chain.append({
                "function_to_use": best_function_name,
                "parameters": parameters
            })
        else:
            # Subsequent functions use the output of the previous function
            result_chain.append({
                "function_to_use": best_function_name,
                "parameters": ["__PREVIOUS_OUTPUT__"]
            })

    return result_chain

# Execute chained functions
def execute_chain(result_chain):
    previous_output = None
    for step in result_chain:
        function_name = step["function_to_use"]
        params = step["parameters"]
        selected_function = function_map[function_name]

        # Replace __PREVIOUS_OUTPUT__ with the actual previous output
        if "__PREVIOUS_OUTPUT__" in params:
            params = [previous_output]

        # Execute the function
        output = selected_function(*params)
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