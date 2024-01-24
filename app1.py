from flask import Flask, request, jsonify
import subprocess
import sys
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import softmax

from flask_cors import CORS  

app = Flask(__name__)
CORS(app)

saved_model_path = 'model'  
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
class_descriptions = [
    "This class refers to the use of time-sensitive language or design elements to create a sense of urgency, often pushing users to take immediate action.",
    "Indicates that the user interface or interaction does not employ any dark patterns. In ethical design, this label suggests transparency and fairness.",
    "Describes tactics that imply limited availability or scarcity of a product or service to encourage quicker decision-making.",
    "Involves guiding users' attention away from important information or actions, often leading them to make unintended choices.",
    "Highlights the use of social influence, testimonials, or statistics to persuade users by showing that others have taken a specific action or made a particular choice.",
    "Refers to deliberate obstacles or challenges in the user journey, hindering smooth navigation or making it difficult for users to complete desired tasks.",
    "Involves covert or deceptive tactics designed to trick users into taking actions they may not have intended to perform.",
    "Implies situations where users are compelled or coerced into taking an action against their will or better judgment."
]
def scrape_and_follow_links(url, depth=1):
    response = requests.get(url)
    data = {"div Tags": []}
    print("hii")
    if response.status_code == 200:
     soup = BeautifulSoup(response.content, 'html.parser')
     tag=['div','p','h1','h2','h3','h4','h5','h6','li']
     c=0
     for i in tag:
        h1_tags = soup.find_all(i)
        for h1_tag in h1_tags:
            if h1_tag:
                k1 = h1_tag.text.strip()
                print(k1)
                user_input = k1
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
                if predicted_label != 'Not Dark Pattern':
                    data["div Tags"].append(h1_tag.text.strip())
                    
                    l=[]
                    l.append(predicted_label)
                    l.append("The page is Deceptive")
                    l.append("Type of deception :")
                    l.append("Description ")
                    for i in range(len(class_labels)):
                        if class_labels[i]== predicted_label:
                            l.append(class_descriptions[i])
                            break
                    l.append("Flaged field" )
                    l.append(h1_tag.text.strip())
                    return l
     l=[]
     l.append("No deception found")
     l.append(" ")
     l.append(" ")
     l.append(" ")
     l.append(" ")
     l.append(" ")
     l.append(" ")
     return l
        
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")

@app.route('/run_python_code', methods=['POST'])
def run_python_code():
    try:
        url = request.json.get('url', '')
        print(url)
        
        k=scrape_and_follow_links(url, depth=2)
        print(k)
        return jsonify({'result': k})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
