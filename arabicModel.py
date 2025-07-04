import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import evaluate

# Load and preprocess dataset
df = pd.read_excel("TweetClassification-Summary.xlsx")
df = df[["text", "aggregatedAnnotation"]]
df = df[df["aggregatedAnnotation"].isin([-1, 0, 1])]
df["label"] = df["aggregatedAnnotation"].replace({-1: 0, 0: 1, 1: 2})

# Split into train/test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

# Convert to Hugging Face dataset format
train_dataset = Dataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()})
test_dataset = Dataset.from_dict({"text": test_texts.tolist(), "label": test_labels.tolist()})
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# Load tokenizer and model

#model_name = "aubmindlab/bert-base-arabertv02"
model_name = "UBC-NLP/MARBERTv2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize dataset
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize)

# Initialize model for 3-class classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Define evaluation metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        "f1_weighted": f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    }

# Set training configuration
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro"
)

# Train and evaluate the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
eval_results = trainer.evaluate()
print(eval_results)
