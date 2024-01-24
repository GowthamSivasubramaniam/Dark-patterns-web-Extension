import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import softmax


saved_model_path = 'model' 
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)


user_input = input("Enter a text for prediction: ")
tokenized_user_input = tokenizer(user_input, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    outputs = model(**tokenized_user_input)
logits = outputs.logits
probs = softmax(logits, dim=-1)
predicted_class_index = torch.argmax(probs, dim=-1).item()
class_labels = ['Urgency', 'Not Dark Pattern', 'Scarcity', 'Misdirection', 'Social Proof', 'Obstruction', 'Sneaking', 'Forced Action'] 
label_encoder = LabelEncoder()
label_encoder.fit(class_labels)
predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]

print(f"User Input: {user_input}")
print(f"Predicted Label: {predicted_label}")
print(f"Predicted Class Probabilities: {probs.tolist()}")
