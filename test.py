import time
import faiss
import spacy
from sentence_transformers import SentenceTransformer


from functions import function_registry
from split_prompt import split_into_actions
from chian_prompts import create_instruction_chainer


tick = time.time()


model = SentenceTransformer('all-MiniLM-L6-v2')



nlp = spacy.load("en_core_web_sm")
doc = nlp("write a poem and then put it in notepad")

tock = time.time()

print(f"Time taken to prepare function embeddings: {tock - tick:.2f} seconds")


actions = split_into_actions(doc)





prompt = "write a poem , required parmeter : prompt , and output type is : str "


function_emdedings = []



# def get_functions_from_actions(actions):
for function_name, function_info in function_registry.items():



    description = function_info.get("description", "")
    required_params = function_info.get("required_params", "")
    output_type = function_info.get("output_type", "")
    param_types = function_info.get("param_types", [])


    if not isinstance(param_types, list):
        param_types = [str(param_types)]


    embed_string = (
        function_name.lower() + " : " +
        description.lower()
        + " parameters required :" + str(required_params).lower()
        + ", output type is : " + str(output_type).lower()
        + " , and param types are : " + ", ".join([str(pt).lower() for pt in param_types])
    )

    function_emdedings.append(embed_string)



    
embeddings = model.encode(function_emdedings, convert_to_numpy=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


query_embedding = model.encode([prompt], convert_to_numpy=True)

# Step 6: Search the FAISS index for the best match (k=1 for the single best match)
k = 1
distances, indices = index.search(query_embedding, k)


texts = [metadata["description"] for metadata in function_registry.values()]
function_names = list(function_registry.keys())


for i, idx in enumerate(indices[0]):

    if idx != -1:  # Skip invalid indices
        print(f"Function: {function_names[idx]}")
        print(f"Description: {texts[idx]} (Distance: {distances[0][i]:.4f})")
    else:
        print(f"No valid match found for rank {i+1} (Distance: {distances[0][i]:.4f})")



# for action in actions:




chain_instructions = create_instruction_chainer()
    

execution_plan_json = chain_instructions(actions)
print(execution_plan_json)