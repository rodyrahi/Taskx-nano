from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load the model
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Prepare training data
train_examples = [
    InputExample(texts=["summarize text using LLM", "Uses LLM to answer questions"], label=0.9),
    InputExample(texts=["summarize it ", "call_llm : uses llm to answer questions parameters required :('prompt',), output type is : str , and param types are : prompt"], label=0.9),
    InputExample(texts=["summarize it", "extract_emails : extract emails from text parameters required :('text',), output type is : str , and param types are : text"], label=0.1),
    InputExample(texts=["search for cat videos", "search a query on google in the default browser parameters required :('query',), output type is : str , and param types are : text"], label=0.95),
    InputExample(texts=["search for cat videos", "call_llm : uses llm to answer questions parameters required :('prompt',), output type is : str , and param types are : prompt"], label=0.1),

    InputExample(texts=["summarize text using LLM", "Extract emails from text"], label=0.1),
    # Add more examples
]

# Create a DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)

# Define the loss function
train_loss = losses.ContrastiveLoss(model=model)

# Fine-tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    warmup_steps=100
)

# Save the fine-tuned model
model.save('fine-tuned-mpnet')