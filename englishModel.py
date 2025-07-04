import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import evaluate
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoConfig,
    BertPreTrainedModel,
    BertModel
)
from datasets import Dataset, DatasetDict
from transformers.modeling_outputs import SequenceClassifierOutput

# Load and preprocess the dataset
df = pd.read_csv("convabuse.csv", sep=";")
df = df[["racist", "sexism", "Input.user"]].dropna()

# Create a single label column from 'racist' and 'sexism'
def determine_label(row):
    if row["racist"] == "racist":
        return "racist"
    elif row["sexism"] == "sexist":
        return "sexist"
    else:
        return "none"

df["label"] = df.apply(determine_label, axis=1)
label_map = {"racist": 0, "sexist": 1, "none": 2}
df["label_id"] = df["label"].map(label_map)

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["Input.user"], df["label_id"], test_size=0.2, stratify=df["label_id"], random_state=42
)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()})
test_dataset = Dataset.from_dict({"text": test_texts.tolist(), "label": test_labels.tolist()})
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# Tokenize text
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
tokenized_dataset = dataset.map(tokenize, batched=True)

# Define custom BERT model with class weights for imbalanced classes
class WeightedBERT(BertPreTrainedModel):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)

# Compute class weights based on label distribution
label_counts = df["label_id"].value_counts().sort_index().values
weights = torch.tensor(1.0 / label_counts, dtype=torch.float)
weights = weights / weights.sum()

# Initialize custom model
config = AutoConfig.from_pretrained(model_name, num_labels=3)
model = WeightedBERT.from_pretrained(model_name, config=config, class_weights=weights)

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
