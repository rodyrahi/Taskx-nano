from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader



# Create training examples
train_examples = [
    # Positive pairs (prompt matches function description)
    InputExample(texts=["open steam application with parameters of types: [PROGRAM]", "Open a software application by name with parameters type : ['PROGRAM_NAME_PARAMETER'] and required parameters : ('text',)"], label=1.0),
    
    InputExample(texts=["open notepad with parameters of types: [PROGRAM]", "Open a software application by name with parameters type : ['PROGRAM_NAME_PARAMETER'] and required parameters : ('text',)"], label=1.0),
    InputExample(texts=["launch discord with parameters of types: [PROGRAM]", "Open a software application by name with parameters type : ['PROGRAM_NAME_PARAMETER'] and required parameters : ('text',)"], label=1.0),



    InputExample(texts=["search google for cats with parameters of types: [URL]", "Perform a web search on a specified engine with parameters type : ['URL_PARAMETER'] and required parameters : ('text',)"], label=1.0),
    
    
    InputExample(texts=["search bing for news with parameters of types: [URL]", "Perform a web search on a specified engine with parameters type : ['URL_PARAMETER'] and required parameters : ('text',)"], label=1.0),
    InputExample(texts=["save file to C:/data.txt with parameters of types: [FILE_PATH]", "Save a file to a specified path with parameters type : ['FILE_PATH_PARAMETER'] and required parameters : ('text',)"], label=1.0),



    # Negative pairs (prompt does not match function description)
    InputExample(texts=["open steam application", "Perform a web search on a specified engine"], label=0.0),
    InputExample(texts=["open steam application", "Save a file to a specified path"], label=0.0),
    InputExample(texts=["search google for cats", "Open a software application by name"], label=0.0),
    InputExample(texts=["search google for cats", "Save a file to a specified path"], label=0.0),
    InputExample(texts=["save file to C:/data.txt", "Open a software application by name"], label=0.0),
    InputExample(texts=["save file to C:/data.txt", "Perform a web search on a specified engine"], label=0.0),
]

# Load the model
model = SentenceTransformer('BAAI/bge-base-en')

# Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define loss function
train_loss = losses.MultipleNegativesRankingLoss(model)

# Fine-tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="fine_tuned_mpnet_model"
)

# Save the model
model.save("fine_tuned_mpnet_model")
print("Model fine-tuning complete and saved to 'fine_tuned_mpnet_model'")