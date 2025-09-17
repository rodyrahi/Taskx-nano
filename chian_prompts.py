from sentence_transformers import SentenceTransformer, util, InputExample, losses
from torch.utils.data import DataLoader
import torch

# Fine-tune the model with synthetic examples
def fine_tune_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    train_examples = [
        InputExample(texts=["generated poem text", "text to put in notepad"], label=0.95),
        InputExample(texts=["poem string", "content for saving"], label=0.9),
        InputExample(texts=["text output", "input text data"], label=0.85),
        InputExample(texts=["image data", "text to save"], label=0.1)  # Negative example
    ]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
    train_loss = losses.CosineSimilarityLoss(model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=10)
    model.save('split_finetuned_model')
    return model

# Load or fine-tune model
try:
    model = SentenceTransformer('split_finetuned_model')
except:
    model = fine_tune_model()

def get_embedding(text):
    return model.encode(text)

def is_compatible(output_desc, input_desc, threshold=0.7):
    if not output_desc or not input_desc:
        return False
    sim = util.cos_sim(get_embedding(output_desc), get_embedding(input_desc))[0][0]
    return sim >= threshold

def infer_output_desc(instruction):
    if "write a poem" in instruction.lower():
        return "generated poem text"
    return "generic output"

def infer_input_desc(instruction):
    if "put it in notepad" in instruction.lower():
        return "text to put in notepad"
    return None

def run_function(instruction, input_data=None):
    """Your function runner - replace with actual."""
    if "write a poem" in instruction:
        return "Roses are red, violets are blue, AI writes poems, just for you."
    elif "put it in notepad" in instruction:
        return f"Saved to notepad: {input_data or 'No content provided'}"
    return "Output"

def chain_instructions(instructions):
    results = []
    last_output = None
    last_output_desc = None
    for i, instr in enumerate(instructions):
        input_desc = infer_input_desc(instr)
        use_input = None
        if i > 0:
            if 'it' in instr.lower() or is_compatible(last_output_desc, input_desc):
                use_input = last_output
        output = run_function(instr, use_input)
        results.append({"instruction": instr, "input": use_input, "output": output})
        last_output = output
        last_output_desc = infer_output_desc(instr)
    return results

# Example
instructions = ["write a email" , "write a poem" , "put it in notepad"]
execution_plan = chain_instructions(instructions)

print("\nExecution Plan:")
for step in execution_plan:
    print(f"Instruction: {step['instruction']}")
    print(f"Input: {step['input'] or 'None'}")
    print(f"Output: {step['output'] or 'None'}")
    print("-" * 50)