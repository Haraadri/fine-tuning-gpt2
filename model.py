from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Load the dataset from a CSV file
csv_path = '/Users/kailash/Downloads/custom_dataset.csv'  # Update with your CSV file path
df = pd.read_csv(csv_path)

# Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)


# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Ensure padding token is set if not already
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Initialize the GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Preprocessing function for dataset
def preprocess_function(examples):
    encoding = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    encoding['labels'] = encoding['input_ids']  # Set the labels to be the same as input_ids
    return encoding

# Apply preprocessing to dataset
dataset = dataset.map(preprocess_function, batched=True)

# Define training arguments for fine-tuning
training_args = TrainingArguments(
    output_dir='./results',            # Output directory
    eval_strategy="epoch",             # Evaluate every epoch
    learning_rate=5e-5,                # Learning rate
    per_device_train_batch_size=2,     # Batch size for training
    per_device_eval_batch_size=2,      # Batch size for evaluation
    num_train_epochs=3,                # Number of training epochs
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('fine_tuned_gpt_model')
tokenizer.save_pretrained('fine_tuned_gpt_model')

