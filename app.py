import streamlit as st
import torch
from data.dataloader import eng_tokenizer, fr_tokenizer
from train import checkpoint_path, device
from models.transformer import build_model
from inference.generate import translate

@st.cache_resource
def load_model():
    model = build_model(device=device)
    checkpoint = torch.load(checkpoint_path, mmap=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

st.title("English to French Translation")

eng_ip = st.text_area("Enter your english input")
max_new_tokens = st.slider("Max New Tokens to Generate", min_value=10, max_value=100)
temperature = st.slider("Temperatue (Controls the Creativity of the Model)", min_value=0.1, max_value=1.5, value=1.0)
do_sample = st.checkbox(label="Sample the next token")

if do_sample == True:
    top_k = st.slider("Top k (Contols the diversity of sampling)", min_value=1, max_value=100)
else:
    top_k = None

submit = st.button("Translate")

if submit:
    model = load_model()
    fr_op = translate(
        model=model,
        eng_ip=eng_ip,
        eng_tokenizer=eng_tokenizer,
        fr_tokenizer=fr_tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_k=top_k
    )
    
    st.subheader("Translated Output:")
    st.write(fr_op)