import sys
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
from sklearn.feature_extraction.text import CountVectorizer
import joblib

model, vectorizer = joblib.load('trained_model_with_vectorizer.joblib')

def scrape_and_follow_links(url, depth=1):
    response = requests.get(url)
    data = {"div Tags": [], "stock": []}
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        h1_tags = soup.find_all('div')
        
        for h1_tag in h1_tags:
            time.sleep(1)
            if h1_tag:
                data["div Tags"].append(h1_tag.text.strip())

                k1 = h1_tag.text.strip()
                print("input:", k1)

                
                user_input = vectorizer.transform([k1])
                y_pred = model.predict(user_input)
                if y_pred[0] == 1:
                    print("Deceptive")
                    break
                else:
                    print("No deception")
        
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
   

if len(sys.argv) == 2:
    url_to_scrape = sys.argv[1]
    scrape_and_follow_links(url_to_scrape, depth=2)
else:
    print("Please provide a URL as a command-line argument.")
