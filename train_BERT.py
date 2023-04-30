import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


# Load data
train_df = pd.read_csv("train.csv")

# Preprocess data
train_df['text'] = train_df['premise'] + " " + train_df['hypothesis']
train_texts = train_df['text'].tolist()
train_labels = train_df['label'].tolist()

# Split data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42)

# Set up tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Tokenize train and validation sets
train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(
    val_texts, truncation=True, padding=True, max_length=128)


# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Create train and validation datasets
train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, val_labels)

# Set up model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=3)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
)

# Set up trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()

# Save the trained model
trainer.model.save_pretrained("./my_trained_model")
