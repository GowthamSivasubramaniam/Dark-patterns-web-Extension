from flask import Flask, request, jsonify
import subprocess
import sys
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
from sklearn.feature_extraction.text import CountVectorizer
import joblib

from flask_cors import CORS  

app = Flask(__name__)
CORS(app)

# Load the model and vectorizer outside of the route
model, vectorizer = joblib.load('trained_model_with_vectorizer.joblib')

def scrape_and_follow_links(url, depth=1):
    response = requests.get(url)
    data = {"div Tags": []}
    print("hii")
    if response.status_code == 200:
     soup = BeautifulSoup(response.content, 'html.parser')
     tag=['li','p','h1','h2','h3','h4','h5','h6','div']
     c=0
     for i in tag:
        h1_tags = soup.find_all(i)
        for h1_tag in h1_tags:
            if h1_tag:
                k1 = h1_tag.text.strip()
                print(k1)
                user_input = vectorizer.transform([k1])
                y_pred = model.predict(user_input)
                if y_pred[0] == 1:
                   c+=1
                   data["div Tags"].append(h1_tag.text.strip())
                   if(c==10):
                    break
        if(c==10):
         break
     if(c>=10):
            with open('output.json', 'w', encoding='utf-8') as json_file:
             json.dump(data, json_file, ensure_ascii=False, indent=4)
            return "Deceptive"
     else:
            return "No deception found"
        
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
