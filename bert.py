import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.nn.functional import softmax
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm 


df = pd.read_csv("dp1.csv")

label_encoder = preprocessing.LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

tokenized_train = tokenizer(train_df['text'].tolist(), padding=True, truncation=True, return_tensors='pt')
tokenized_val = tokenizer(val_df['text'].tolist(), padding=True, truncation=True, return_tensors='pt')


train_dataset = TensorDataset(tokenized_train['input_ids'], tokenized_train['attention_mask'], torch.tensor(train_df['label'].tolist()))
val_dataset = TensorDataset(tokenized_val['input_ids'], tokenized_val['attention_mask'], torch.tensor(val_df['label'].tolist()))

batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()


num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0  
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

    for batch in progress_bar:
        input_ids, attention_mask, label = batch
        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=label)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        progress_bar.set_postfix({'training_loss': train_loss / len(train_dataloader)})


    model.eval()
    val_preds = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc='Validation', leave=False):
            input_ids, attention_mask, label = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = softmax(logits, dim=-1)
            val_preds.extend(torch.argmax(probs, dim=-1).tolist())


    val_labels = val_df['label'].tolist()
    accuracy = accuracy_score(val_labels, val_preds)
    print(f"Epoch {epoch + 1}/{num_epochs} - Validation Accuracy: {accuracy * 100:.2f}%")


model.save_pretrained('model')
