import torch
import re
import json
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments, Trainer
from datasets.arrow_dataset import Column
print(torch.cuda.is_available())  # Should return True
print(torch.__version__)  # Verify PyTorch version

# Replace "squad" with your dataset name

model_name = "huggyllama/llama-7b"

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = text.lower().strip()  # Lowercase and strip whitespace
    return text

def preprocess_data(input_file, output_file):
    processed_data = []
    with open(input_file, "r") as f:
        for line in f:
            # Assuming each line is plain text
            input_text = clean_text(line.strip())
            output_text = generate_response(input_text)  # Define your logic here
            processed_data.append({"input": input_text, "output": output_text})

    with open(output_file, "w") as f:
        for entry in processed_data:
            f.write(json.dumps(entry) + "\n")


tokenizer = LlamaTokenizer.from_pretrained(model_name)
dataset = load_dataset("squad")
tokenized_data = tokenizer.tokenize(
    dataset["train"]["question"][0],  # Replace "context" with your text column
    truncation=True,
    padding=True,
    max_length=512
)

print(tokenized_data)  # Check the first 5 tokenized inputs

# Load the pretrained LLaMA model
print("Loading model...")
model = LlamaForCausalLM.from_pretrained(model_name)
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

# Freeze the base model layers
for param in model.base_model.parameters():
    param.requires_grad = False

# Configure training arguments
training_args = TrainingArguments(
    output_dir="./llama_finetuned",
    save_steps=1000,
    logging_steps=500,
    per_device_train_batch_size=4,  # Adjust based on your GPU memory
    gradient_accumulation_steps=8,  # Effective batch size = 4 x 8
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_dir="./logs",
    save_total_limit=2  # Keeps only the last two checkpoints
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset
)

# Start training
trainer.train()
import wandb

wandb.init(project="llama-finetune", name="experiment_1")

training_args.report_to = ["wandb"]
import wandb

wandb.init(project="llama-finetune", name="experiment_1")

training_args.report_to = ["wandb"]

from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize FastAPI
app = FastAPI()

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-30b_finetuned")
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-30b_finetuned")

@app.post("/generate")
def generate(prompt: str):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate output
    outputs = model.generate(inputs["input_ids"], max_length=100)
    
    # Decode and return the generated text
    return {"generated_text": tokenizer.decode(outputs[0])}

import nlpaug.augmenter.word as naw
# Synonym-based augmentation
augmenter = naw.SynonymAug()
augmented_text = augmenter.augment("This is a sample sentence.")
