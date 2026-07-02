from inference import inference_llm_jepa
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from collator import DataCollatorForLLMJEPA
from model import LLM_JEPA
from train import train_model
from losses import LLM_JEPALoss
import torch
from evaluate import evaluate_step

# Load a large and diverse text dataset, e.g., 'wikitext-103-raw-v1'
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')

print(dataset)
# Instantiate a tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
block_size=128
# Define a block size for the tokenized chunks
# This is a common practice for language modeling to create fixed-size sequences
def tokenize_function(examples):
    # Concatenate all texts in the batch
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

def preprocess_function(examples):
    # First, tokenize individual texts, handle short texts by filtering them out later
    tokenized_batch = tokenizer(examples['text'], truncation=True, max_length=tokenizer.model_max_length)
    return tokenized_batch

# Map the preprocess function first to tokenize individual examples
tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=4, remove_columns=['text'])

# Filter out examples that are too short after initial tokenization, e.g., only [CLS] [SEP]
tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) > 2, num_proc=4)

# Now apply the chunking function
tokenized_dataset = tokenized_dataset.map(tokenize_function, batched=True, num_proc=4)


# Instantiate the custom data collator
data_collator = DataCollatorForLLMJEPA(
    tokenizer=tokenizer,
    mlm_probability=0.15,
    mask_token_id=tokenizer.mask_token_id,
    pad_token_id=tokenizer.pad_token_id
)

# Create DataLoaders for training and validation
train_dataloader = DataLoader(
    tokenized_dataset["train"],
    shuffle=True,
    batch_size=64, # Adjust batch size as needed
    collate_fn=data_collator,
    num_workers=2 # Adjust num_workers for faster data loading
)

val_dataloader = DataLoader(
    tokenized_dataset["validation"],
    batch_size=64, # Adjust batch size as needed
    collate_fn=data_collator,
    num_workers=2 # Adjust num_workers for faster data loading
)

# 1. Define model hyperparameters
vocab_size = tokenizer.vocab_size
d_model = 256 # Embedding dimension
num_layers = 4 # Number of encoder layers
num_heads = 8 # Number of attention heads
dim_feedforward = 1024 # Dimension of the feed-forward network in Transformer blocks
max_len = block_size # Max sequence length, consistent with data preparation
predictor_hidden_dim = 512 # Hidden dimension for the predictor MLP
predictor_num_layers = 2 # Number of layers in the predictor MLP
dropout = 0.1 # Dropout rate


# 2. Instantiate the LLM_JEPA model
model = LLM_JEPA(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    dim_feedforward=dim_feedforward,
    max_len=max_len,
    predictor_hidden_dim=predictor_hidden_dim,
    predictor_num_layers=predictor_num_layers,
    dropout=dropout
)

print("LLM_JEPA model instantiated.")

# 3. Define an optimizer for the online encoder and predictor
# We combine parameters from both online_encoder and predictor
optimizer = optim.AdamW(
    list(model.online_encoder.parameters()) + list(model.predictor.parameters()),
    lr=1e-4
)

print("Optimizer (AdamW) initialized for online_encoder and predictor.")

# 4. Optionally, define a learning rate scheduler
# Assuming a total number of training steps (e.g., based on dataset size and epochs)
# For demonstration, let's estimate T_max based on train_dataloader length
T_max = len(train_dataloader) * 10 # 10 epochs
scheduler = CosineAnnealingLR(optimizer, T_max=T_max)

print(f"Learning rate scheduler (CosineAnnealingLR with T_max={T_max}) initialized.")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model moved to: {device}")
# Define training parameters
n_epochs = 5
momentum_rate = 0.99 # Momentum for target network update


untrained_val_loss = evaluate_step(
    model=model,
    dataloader=val_dataloader,
    loss_fn=LLM_JEPALoss(),
    device=device
)

train_model(n_epochs,model,train_dataloader,val_dataloader,optimizer,scheduler,momentum_rate,device)

print("\n--- Demonstrating usage of inference_llm_jepa ---")

sample_text = "This is a sample sentence to test the LLM-JEPA model's inference capabilities."

# Ensure model is on the correct device
model.to(device)

embeddings = inference_llm_jepa(
    model=model,
    tokenizer=tokenizer,
    text=sample_text,
    device=device,
    block_size=block_size
)

print(f"Sample input text: '{sample_text}'")
print(f"Shape of extracted embeddings: {embeddings.shape}")
print(f"First 5 embedding dimensions of the first token:\n{embeddings[0, 0, :5]}")