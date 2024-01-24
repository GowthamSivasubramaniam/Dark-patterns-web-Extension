import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import softmax
import torch
import joblib

# Load your dataset (replace 'your_dataset.csv' with your actual file)
df = pd.read_csv('data.csv')
df1 = pd.read_csv('data1.csv')

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df,df1, test_size=0.2, random_state=42)

# Load pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and pad the training text
train_texts = train_df['Text'].tolist()
train_labels = train_df['Label'].tolist()
train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors='pt')

# Convert labels to tensor
train_labels = torch.tensor(train_labels)

# Create a DataLoader for training
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Fine-tune the model (this is a simplified example, actual fine-tuning requires more steps)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_epochs = 3

for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save the fine-tuned model
model.save_pretrained('fine_tuned_model')

# Tokenize and pad the test text
test_texts = test_df['Text'].tolist()
test_encodings = tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')

# Inference on the test set
with torch.no_grad():
    test_outputs = model(**test_encodings)
    logits = test_outputs.logits

# Convert logits to probabilities using softmax
probs = softmax(logits, dim=1)

# Get predicted labels (0 or 1 in binary classification)
predicted_labels = torch.argmax(probs, dim=1).tolist()
print(predicted_labels)
