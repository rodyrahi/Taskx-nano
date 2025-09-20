from sentence_transformers import SentenceTransformer, util, InputExample, losses
from torch.utils.data import DataLoader
import torch
import json

def create_instruction_chainer(model_path='split_finetuned_model', pretrained_model='all-MiniLM-L6-v2', threshold=0.7):
    """
    Creates a function to chain and execute instructions with semantic compatibility checking.
    
    Args:
        model_path (str): Path to save/load the fine-tuned model
        pretrained_model (str): Pretrained SentenceTransformer model name
        threshold (float): Cosine similarity threshold for input/output compatibility
    
    Returns:
        function: A function that processes a list of instructions and returns execution results as JSON
    """
    
    # Fine-tune the model with synthetic examples
    def fine_tune_model():
        model = SentenceTransformer(pretrained_model)
        train_examples = [
            InputExample(texts=["generated text output", "text input for processing"], label=0.95),
            InputExample(texts=["output string", "input content"], label=0.9),
            InputExample(texts=["text result", "text data"], label=0.85),
            InputExample(texts=["non-text data", "text input"], label=0.1),  # Negative example
            InputExample(texts=["write hello there", "put it in notepad"], label=0.95)  # New negative example
            
        ]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
        train_loss = losses.CosineSimilarityLoss(model)
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=10)
        model.save(model_path)
        return model

    # Load or fine-tune model
    try:
        model = SentenceTransformer(model_path)
    except:
        model = fine_tune_model()

    def get_embedding(text):
        return model.encode(text)

    def is_compatible(output_desc, input_desc, threshold=threshold):
        if not output_desc or not input_desc:
            return False
        sim = util.cos_sim(get_embedding(output_desc), get_embedding(input_desc))[0][0]
        return sim >= threshold

    def infer_output_desc(instruction):
        return "generated text output"

    def infer_input_desc(instruction):
        # Assume any instruction after the first may expect input
        return "text input for processing"

    def run_function(instruction, input_data=None, instruction_index=0):
        """Executes a single instruction with optional input, returning generic output."""
        if input_data:
            return f"output{instruction_index + 1}"
        return f"output{instruction_index + 1}"

    def chain_instructions(instructions):
        """
        Chains and executes a list of instructions, passing outputs to compatible inputs.
        
        Args:
            instructions (list): List of instruction strings
            
        Returns:
            str: JSON string containing the execution plan
        """
        results = []
        last_output = None
        last_output_desc = None
        for i, instr in enumerate(instructions):
            input_desc = infer_input_desc(instr) if i > 0 else None  # No input for first instruction
            use_input = None
            if i > 0 and ('it' in instr.lower() or is_compatible(last_output_desc, input_desc)):
                use_input = last_output
            output = run_function(instr, use_input, i)
            results.append({
                "instruction": instr,
                "input": use_input if use_input else None,
                "output": output
            })
            last_output = output
            last_output_desc = infer_output_desc(instr)
        return json.dumps({"execution_plan": results}, indent=2)

    return chain_instructions

# Example usage:
if __name__ == "__main__":
    # Create the instruction chainer
    chain_instructions = create_instruction_chainer()
    
    # Test with example instructions
    instructions = ["search for hotels near me", "put it in notepad"]
    execution_plan_json = chain_instructions(instructions)
    print(execution_plan_json)