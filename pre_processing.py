import requests
import re
import json

def fetch_and_clean_data(url):
    response = requests.get(url)
    response.raise_for_status()  
    lines = response.text.split('\n') 

    cleaned_lines = []
    for line in lines:
        cleaned_line = line.strip().lower() 
        if cleaned_line:  
            cleaned_line = re.sub(r'[^a-zA-Z0-9 \.]', '', cleaned_line) 
            cleaned_lines.append(cleaned_line)

    return cleaned_lines

url = 'https://www.gutenberg.org/files/1661/1661-0.txt'
cleaned_lines = fetch_and_clean_data(url)

text = ' '.join(cleaned_lines)
words = text.split()

unique_words = sorted(set(words))
stoi = {s: i + 1 for i, s in enumerate(unique_words)}
stoi['.'] = 0  
itos = {i: s for s, i in stoi.items()}  

with open(r"C:\Users\sivak\Desktop\ML3\stoi.json", "w") as f:
    json.dump(stoi,f)

with open(r"C:\Users\sivak\Desktop\ML3\itos.json", "w") as f:
    json.dump(itos,f)


print("Vocabulary mappings saved to 'stoi.json' and 'itos.json'")
