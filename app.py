import streamlit as st
import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

with open(r"C:\Users\sivak\Desktop\ML3\stoi.json", "r") as f:
    stoi = json.load(f)

with open(r"C:\Users\sivak\Desktop\ML3\itos.json", "r") as f:
    itos = json.load(f)
    
def preprocess_input(input_text, stoi, block_size=15):
    input_words = input_text.strip().lower().split()
    input_indices = [stoi.get(word, stoi['.']) for word in input_words]
    input_indices = input_indices[-block_size:]
    input_tensor = [0] * (block_size - len(input_indices))+ input_indices 
    return torch.tensor(input_tensor[-block_size:], dtype=torch.long).unsqueeze(0)

def decode_index(indices):
    return [itos[str(i.item())] for i in indices]

def predict_next_words(input_text, k):
    input_tensor = preprocess_input(input_text, stoi, block_size)
    model.eval()
    predicted_words = input_text+" "

    for _ in range(k):
        with torch.no_grad():
            output = model(input_tensor) 
            top_k_indices = output.topk(1).indices[0] 
            predicted_word = decode_index(top_k_indices)
            predicted_words+=(predicted_word[0]+" ")
            
            new_input = input_tensor[0].tolist()[1:] + top_k_indices.tolist() 
            input_tensor = torch.tensor(new_input, dtype=torch.long).unsqueeze(0).to(input_tensor.device)  

    return predicted_words

class NextWord(torch.nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, activation_function='relu', hidden_size=1024):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, emb_dim)
        self.lin1 = torch.nn.Linear(block_size * emb_dim, hidden_size)
        
        if activation_function == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation_function == 'tanh':
            self.activation = torch.nn.Tanh()
        self.lin2 = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = self.activation(self.lin1(x))
        x = self.lin2(x)
        return x

def load_model(emb_dim, block_size, activation_function):
    model = NextWord(block_size=block_size, vocab_size=len(stoi), emb_dim=emb_dim, activation_function=activation_function)
    model_path = f"C:\\Users\\sivak\\Desktop\\ML3\\model_emb_{emb_dim}_block_{block_size}_{activation_function}.pth"  # Adjust path as needed
    model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    model.eval()
    return model

st.title("Next Word Prediction App")

emb_dim_options = [32, 64] 
block_size_options = [5, 10, 15]  
activation_options = ['relu', 'tanh'] 

emb_dim = st.selectbox("Select Embedding Size:", emb_dim_options)
block_size = st.selectbox("Select Context Length:", block_size_options)
activation_function = st.selectbox("Select Activation Function:", activation_options)

model = load_model(emb_dim, block_size, activation_function)

user_input = st.text_input("Enter your input text:")
k = st.number_input("Enter the number of predictions (k):", min_value=1, value=5)

if st.button("Predict"):
    complete_sentence = user_input
    complete_sentence= predict_next_words(complete_sentence, k)
    st.write("Predicted complete sentence:")
    st.write(complete_sentence)

def get_visualization_path(emb_dim, block_size, activation_function):
    if activation_function == 'relu':
        return f"C:\\Users\\sivak\\Desktop\\ML3\\emb_{emb_dim}_relu_{block_size}.png"
    elif activation_function == 'tanh':
        return f"C:\\Users\\sivak\\Desktop\\ML3\\emb_{emb_dim}_tanh_{block_size}.png"
    return None

if st.button("Visualize t-SNE Embeddings"):
    tsne_image_path = get_visualization_path(emb_dim, block_size, activation_function)
    if tsne_image_path and os.path.exists(tsne_image_path):
        st.image(tsne_image_path, caption="t-SNE Visualization of Word Embeddings", use_column_width=True)
    else:
        st.write("Visualization image not found.")

if st.button("Visualize Base Model"):
    base_model_image_path = r"C:\Users\sivak\Desktop\ML3\Base_model.png"
    st.image(base_model_image_path, caption="Base Model Visualization", use_column_width=True)
