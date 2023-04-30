import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def main():
    model_path = "./my_trained_model/"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    test_df = pd.read_csv("test.csv")

    test_df['text'] = test_df['premise'] + " " + test_df['hypothesis']
    test_texts = test_df['text'].tolist()

    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True, max_length=128)

    test_labels = np.zeros(len(test_texts))

    test_dataset = CustomDataset(test_encodings, test_labels)

    training_args = TrainingArguments(
        per_device_eval_batch_size=64,
        output_dir="."
    )

    trainer = Trainer(
        model=model,
        args=training_args,
    )

    test_preds = trainer.predict(test_dataset)

    test_labels = np.argmax(test_preds.predictions, axis=1)

    submission_df = pd.DataFrame(
        {"id": test_df["id"], "prediction": test_labels})

    submission_df.to_csv("submission.csv", index=False)

if __name__ == '__main__':
    main()
