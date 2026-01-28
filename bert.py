
# this code is transformer

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
from collections import Counter
import matplotlib.pyplot as plt
import nltk

# Load dataset
data = pd.read_csv('/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')

# Fill missing text columns and combine them
text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
data[text_columns] = data[text_columns].fillna('')
data['text'] = data[text_columns].apply(lambda x: ' '.join(x), axis=1)

# Preprocess text minimally
def preprocess_text(text):
    return text.replace('\n', ' ').strip()

data['text'] = data['text'].apply(preprocess_text)

# Target variable
data = data.dropna(subset=['fraudulent'])  # Ensure no missing target labels
data['fraudulent'] = data['fraudulent'].astype(int)

# Split dataset into train and test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['text'], data['fraudulent'], test_size=0.2, random_state=42, stratify=data['fraudulent']
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=256)

# Prepare datasets
train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'], 'labels': train_labels.tolist()})
test_dataset = Dataset.from_dict({'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask'], 'labels': test_labels.tolist()})

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    report_to="none"
)

# Define evaluation metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {'accuracy': accuracy}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=-1)

# Classification report
print("\nClassification Report:\n", classification_report(test_labels, y_pred))

# Confusion matrix
cm = confusion_matrix(test_labels, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraudulent', 'Fraudulent'], yticklabels=['Not Fraudulent', 'Fraudulent'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
y_proba = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
fpr, tpr, thresholds = roc_curve(test_labels, y_proba)
auc_score = roc_auc_score(test_labels, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


fraudulent_text = data[data['fraudulent'] == 1]['clean_text'].tolist()
all_words = ' '.join(fraudulent_text).split()

# Count word frequencies
word_counts = Counter(all_words)

# Get the top 20 most common words
top_words = word_counts.most_common(20)

# Display the top 20 words and their frequencies
print("Top 20 Words in Fraudulent Job Posts:")
for word, count in top_words:
    print(f"{word}: {count}")

# Plot the top 20 words
words, counts = zip(*top_words)
plt.figure(figsize=(12, 6))
plt.barh(words, counts, color='skyblue')
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.title("Top 20 Words in Fraudulent Job Posts")
plt.gca().invert_yaxis()  # Invert y-axis to have the highest frequency word at the top
plt.show()
